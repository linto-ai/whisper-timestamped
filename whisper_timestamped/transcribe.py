#!/usr/bin/env python3

__author__ = "Jérôme Louradour"
__credits__ = ["Jérôme Louradour"]
__license__ = "MIT"

# Whisper and Torch
import whisper
import torch
import torch.nn.functional as F

# For alignment
import numpy as np
import dtw
# from scipy.signal import medfilt as median_filter
from scipy.ndimage import median_filter # faster owing to https://github.com/openai/whisper/commit/f0083e7eb20d032390e42f6f6039947fa8669c93

# Additional
import string
import csv

# Constant variables
from whisper.utils import format_timestamp
from whisper.audio import N_FRAMES, HOP_LENGTH, SAMPLE_RATE  # 3000, 160, 16000
AUDIO_SAMPLES_PER_TOKEN = HOP_LENGTH * 2                     # 320
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / SAMPLE_RATE # 0.02

# Logs
import logging
logger = logging.getLogger("whisper_timestamped")

def transcribe_timestamped(
    # main Whisper options
    model,
    audio,
    language=None,
    task="transcribe",
    # additional options for word alignment
    remove_punctuation_from_words=False,
    compute_word_confidence=True,
    include_punctuation_in_confidence=False,
    refine_whisper_precision=0.5,
    min_word_duration=0.04,
    plot_word_alignment=False,
    seed=1234,
    # other Whisper options
    temperature=0.0, # TODO: support list
    best_of=None,
    beam_size=None, # TODO: support 5
    patience=None,
    length_penalty=None,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
    fp16=None,
    condition_on_previous_text=True,
    initial_prompt=None,
    suppress_tokens="-1",
    sample_len=None,
    verbose=False,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance.

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform.

    language: str
        The language to use for the transcription. If None, the language is detected automatically.

    task: str
        The task to perform: either "transcribe" or "translate".

    remove_punctuation_from_words: bool
        If False, words will be glued with the next punctuation mark (if any).
        If True, there will be no punctuation mark in the `words[:]["text"]` list.
        It only affects these strings; This has no influence on the computation of the word confidence, whatever the value of `include_punctuation_in_confidence` is.

    compute_word_confidence: bool
        Whether to compute word confidence.
        If True, a finer confidence for each segment will be computed as well.
    
    include_punctuation_in_confidence: bool
        Whether to include proba of punctuation in the computation of the (previous) word confidence.

    refine_whisper_precision: float
        How much can we refine Whisper segment positions, in seconds. Must be a multiple of 0.02.

    min_word_duration: float
        Minimum duration of a word, in seconds. If a word is shorter than this, timestamps will be adjusted.

    plot_word_alignment: bool
        Whether to plot the word alignment for each segment. matplotlib must be installed to use this option.

    seed: int
        Random seed to use for temperature sampling, for the sake of reproducibility.
        Choose None for unpredictable randomness.

    temperature: float
        Temperature for sampling.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed.

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed.

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent.

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    initial_prompt: str
        Optional text to provide as a prompt for the first window.

    suppress_tokens: str
        Comma-separated list of token ids to suppress during sampling;
        '-1' will suppress most special characters except common punctuations.

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    debug = logger.getEffectiveLevel() >= logging.DEBUG

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Check input options
    assert refine_whisper_precision >= 0 and refine_whisper_precision / AUDIO_TIME_PER_TOKEN == round(refine_whisper_precision / AUDIO_TIME_PER_TOKEN), f"refine_whisper_precision must be a positive multiple of {AUDIO_TIME_PER_TOKEN}"
    refine_whisper_precision_nsamples = round(refine_whisper_precision / AUDIO_TIME_PER_TOKEN)
    assert min_word_duration >= 0, f"min_word_duration must be a positive number"

    if isinstance(temperature, (list, tuple)) and len(temperature) > 1:
        raise NotImplementedError("Transcription with several temperatures not implemented")
    if beam_size is not None:
        raise NotImplementedError("Transcription with beam search not implemented")

    # Input options
    if fp16 is None:
        fp16 = model.device != torch.device("cpu")

    def get_logit_filters(prompt):
        decoding_options = whisper.DecodingOptions(
            task=task,
            language=language,
            temperature=temperature,
            sample_len=sample_len,
            best_of=best_of,
            beam_size=beam_size,
            patience=patience,
            length_penalty=length_penalty,
            prompt=prompt,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=suppress_tokens,
            without_timestamps=False,
            max_initial_timestamp=1.0,
            fp16=fp16,
        )
        # This performs some checks on the options
        decoding_task = whisper.decoding.DecodingTask(model, decoding_options)
        return decoding_task.logit_filters
    
    logit_filters = get_logit_filters(initial_prompt)
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, task=task, language=language)
    max_sample_len = sample_len or model.dims.n_text_ctx // 2 

    # Safety check
    input_stride = N_FRAMES // model.dims.n_audio_ctx
    time_precision = input_stride * HOP_LENGTH / SAMPLE_RATE
    assert time_precision == AUDIO_TIME_PER_TOKEN

    # Note: we cannot trust the token in the middle of tokenizer.sot_sequence which refers to the language
    #       (arbitrarily set to <|en|> if it's actually None/unknown)
    token_sot = tokenizer.sot
    token_eot = tokenizer.eot

    use_space = whisper.tokenizer.TO_LANGUAGE_CODE.get(str(language).lower(), language) not in ["zh", "ja", "th", "lo", "my"]

    # The main outcome
    timestamped_word_segments = []  # list of timestamped word segments that have been collected so far
    # Main variables to be accumulated
    segment_tokens = [[]]              # list of lists of token indices that have been collected so far (one list per segment)
    segment_attweights = [[] for _ in range(len(model.decoder.blocks))]
                                    # attention weights on the last segments
    segment_avglogprobs = []        # average log probability for each segment (actually of the corresponding chunk, as computed by whisper)
    segment_logprobs = []           # token log probabilities for each segment
    # Variables related to options that can skip some segments
    sot_index = None                # index of the SOT token in the current set of processed tokens
    no_speech_prob = None           # no speech probability for the current 30 sec chunk
    chunk_logprobs = []             # log probabilities for the current 30 sec chunk
    chunk_tokens = []               # tokens for the current 30 sec chunk (list of Torch tensors)
    chunk_tokens_nosot = []         # tokens for the current 30 sec chunk, without the SOT tokens (list of indices)
    last_token_fallback = None      # last token to use as a fallback if the model gets stuck
    has_started = False             # whether we have started decoding
    mfcc = None                     # MFCC features for the current 30 sec chunk
    num_inference_steps = 0         # number of inference steps performed so far (for debugging only)

    def reset(add_segment, keep_last_token):
        """ Reset the list of tokens for the current speech segment, and corresponding cross-attention weights """
        nonlocal segment_tokens, segment_attweights
        if add_segment:
            if keep_last_token:
                segment_tokens.append([segment_tokens[-1][-1]])
                segment_attweights = [w[-1:] for w in segment_attweights]
            else:
                segment_tokens.append([])
                segment_attweights = [[] for w in segment_attweights]
            segment_tokens[-2].pop(0)
            if debug:
                logger.debug(f"Added new segment: {tokenizer.decode_with_timestamps(segment_tokens[-2])}")
        elif len(segment_tokens[-1]) > 0:
            segment_tokens[-1] = []
            segment_attweights = [[] for w in segment_attweights]
        if debug:
            logger.debug(f"Reset last segment to: {tokenizer.decode_with_timestamps(segment_tokens[-1])}")

    saw_consecutive_timestamps = True
    def must_flush_segment(curr_tokens):
        """ Return whether or not the previously collected tokens must be used to add a new speech segment """
        nonlocal segment_tokens, saw_consecutive_timestamps, chunk_tokens_nosot
        if curr_tokens is not None and len(curr_tokens) == 1:
            is_timestamp = curr_tokens[0] >= tokenizer.timestamp_begin
            is_previous_timestamp = segment_tokens[-1][-1] >= tokenizer.timestamp_begin if len(segment_tokens[-1]) > 0 else False
            consecutive_timestamps = is_timestamp and is_previous_timestamp
            if consecutive_timestamps:
                saw_consecutive_timestamps = True
            if len(chunk_tokens_nosot) == max_sample_len - 2 and is_timestamp:
                consecutive_timestamps = True
            return consecutive_timestamps
        else: # Several tokens as a prompt or must flush last segments
            must_flush = not saw_consecutive_timestamps
            logger.debug(f"New prompt: flushing = {must_flush}")
            if not must_flush:
                # Discard the end of the last transcription
                reset(False, True)
            saw_consecutive_timestamps = False
            return must_flush

    index_begin_30sec_chunck = 0
    def get_index_begin_30sec_chunck(curr_tokens):
        nonlocal index_begin_30sec_chunck

        if curr_tokens is None or len(curr_tokens) > 1:
            res = index_begin_30sec_chunck
            index_begin_30sec_chunck = len(segment_tokens)-1
            return res

    def may_flush_segment(curr_tokens = None):
        """ Add a speech segment with the new tokens if necessary.
            May also remove the last collected segments if filtered out by Whisper (no_speech_prob <= no_speech_threshold)
        """
        nonlocal segment_tokens, segment_attweights, timestamped_word_segments, has_started, no_speech_prob, chunk_tokens, chunk_tokens_nosot, chunk_logprobs, mfcc, logit_filters, index_begin_30sec_chunck, last_token_fallback, num_inference_steps

        # Check if a new segment should be added
        unfinished_decoding = False
        if must_flush_segment(curr_tokens):

            if debug:
                logger.debug(f"Adding segment {len(timestamped_word_segments)+1} at step {num_inference_steps}:\n\t{tokenizer.decode_with_timestamps(segment_tokens[-1])}")
            tokens = segment_tokens[-1][1:]
            # When the decoding hit the max limit (number of tokens) -- usually when the language model gets stuck --
            # then we have to recover the last token from what is send to the decoder
            unfinished_decoding = len(tokens) and tokens[-1] < tokenizer.timestamp_begin
            last_token_reliable = True
            if unfinished_decoding:
                # The last token chosen is in the prompt for the new chunk
                if curr_tokens is not None and curr_tokens[0] == tokenizer.sot_prev:
                    last_token_fallback = curr_tokens[-4].item()
                # Fallback for the last segment, or without prompt: Assume greedy decoding
                else:
                    last_token_fallback = torch.argmax(chunk_logprobs[-1]).item()
                    last_token_reliable = (temperature == 0)
                if debug:
                    logger.debug(f"WARNING: also add last token: {tokenizer.decode_with_timestamps([last_token_fallback])}")
                tokens.append(last_token_fallback)
                segment_tokens[-1].append(last_token_fallback)
                attention_weights = [torch.cat(w, dim=-2) for w in segment_attweights]
            else:
                attention_weights = [torch.cat(w[:-1], dim=-2) for w in segment_attweights]
            ws = perform_word_alignment(
                tokens,
                attention_weights,
                tokenizer,
                use_space=use_space,
                refine_whisper_precision_nsamples=refine_whisper_precision_nsamples,
                mfcc=mfcc,
                plot=plot_word_alignment,
                remove_punctuation_from_words=remove_punctuation_from_words,
                unfinished_decoding=unfinished_decoding,
            )
            add_segment = len(ws) > 0
            if add_segment:
                timestamped_word_segments.append(ws)
                # Add info about the dedoding that did not go well...
                if unfinished_decoding:
                    timestamped_word_segments[-1][-1]["avg_logprob_reliable"] = last_token_reliable
            else:
                logger.debug(f"Not added!")
            reset(add_segment, curr_tokens is not None and len(curr_tokens) == 1)

        i_start = get_index_begin_30sec_chunck(curr_tokens)
        # All segments from previous 30sec chunck have been collected
        if (i_start is not None and has_started):

            # Get word confidence and/or check if previous segments shoud have been skipped
            if compute_word_confidence or no_speech_threshold is not None:

                # no voice activity check
                should_skip = (no_speech_prob > no_speech_threshold) if (no_speech_threshold is not None) else False
                if compute_word_confidence or (should_skip and logprob_threshold is not None):
                    n = len(chunk_logprobs)
                    if unfinished_decoding:
                        assert last_token_fallback is not None
                        last_tokens = [last_token_fallback]
                        n += 1
                    elif len(chunk_tokens_nosot) == max_sample_len - 1:
                        # there were segments in the 30sec chunck, and then the LM got stuck
                        last_tokens = [torch.argmax(chunk_logprobs[-1]).item()]
                        timestamped_word_segments[-1][-1]["avg_logprob_reliable"] = (temperature == 0)
                    else:
                        last_tokens = [tokenizer.eot]
                    chunck_indices = chunk_tokens_nosot + last_tokens
                    assert len(chunk_logprobs) == len(chunck_indices), f"{len(chunk_logprobs)} != {len(chunck_indices)}"
                    logprobs = torch.cat([logprob[i].unsqueeze(0) for (logprob, i) in zip(chunk_logprobs, chunck_indices)])
                    assert min([p.isfinite().item() for p in logprobs]), "Got infinite logprob"
                    sum_logprob = sum(logprobs)
                    avg_logprob = sum_logprob/n
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    if avg_logprob > logprob_threshold:
                        should_skip = False
                
                if should_skip:
                    logger.debug(f"Skipping last {len(segment_tokens)-1-i_start} segments (no_speech_prob {no_speech_prob} > {no_speech_threshold} and avg_logprob {avg_logprob} < {logprob_threshold})")
                    index_begin_30sec_chunck -= len(segment_tokens)-1-i_start
                    segment_tokens = segment_tokens[:i_start] + [segment_tokens[-1]]
                    timestamped_word_segments = timestamped_word_segments[:i_start]
                elif compute_word_confidence:
                    avg_logprob = avg_logprob.item()
                    i_token_end = -1
                    for i in range(i_start, len(segment_tokens)-1):
                        tokens = segment_tokens[i]
                        i_token_start = i_token_end + 1
                        i_token_end = i_token_start + len(tokens)
                        assert chunck_indices[i_token_start:i_token_end] == tokens, f"Inconsistent token list {tokenizer.decode_with_timestamps(chunck_indices[i_token_start:i_token_end])} != {tokenizer.decode_with_timestamps(tokens)}"
                        i_token_start += 1 # skip sos (start time)
                        if not unfinished_decoding:
                            i_token_end -= 1 # skip eos (end time)
                        segment_logprobs.append(logprobs[i_token_start:i_token_end])
                        segment_avglogprobs.append(avg_logprob)
            else:
                for i in range(i_start, len(segment_tokens)-1):
                    segment_logprobs.append(None)
                    segment_avglogprobs.append(None)

            # Reset counters
            chunk_tokens = []
            chunk_tokens_nosot = []
            chunk_logprobs = []
            no_speech_prob = None

    def hook_attention_weights(layer, ins, outs, index):
        nonlocal segment_attweights
        # In old version of whisper, output is a single tensor
        assert isinstance(outs, tuple) and len(outs) == 2, "whisper seems to be outdated, please update it (pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git)"
        w = outs[-1]
        # Only the last attention weights is useful
        if w.shape[-2] > 1:
            w = w[:, :, -1:, :]
        segment_attweights[index].append(w)

    def hook_mfcc(layer, ins, outs):
        nonlocal mfcc
        mfcc = ins[0]

    def hook_input_tokens(layer, ins, outs):
        nonlocal segment_tokens, sot_index, chunk_tokens, chunk_tokens_nosot, logit_filters, has_started, language, num_inference_steps
        num_inference_steps += 1
        curr_tokens = ins[0].squeeze(0)

        if len(curr_tokens) > 1:
            chunk_prompt = curr_tokens.tolist()
            if not has_started and language is None:
                language = tokenizer.decode(curr_tokens[1:2])[2:-2]
            logit_filters = get_logit_filters(chunk_prompt[1:-3])
        
        may_flush_segment(curr_tokens)

        # Keep the last token only
        segment_tokens[-1].append(curr_tokens[-1].item())

        # Get the index of the <|startoftranscript|> tokens (to get proba of silence later)
        if len(curr_tokens) > 1:
            has_started = True
            if no_speech_threshold is not None:
                sot_index = curr_tokens.tolist().index(tokenizer.sot)
        else:
            sot_index = None

        # Accumulate tokens
        if has_started:
            chunk_tokens.append(curr_tokens)
            if len(curr_tokens) == 1:
                chunk_tokens_nosot.append(curr_tokens[-1].item())

    embedding_weights = None 
    def hook_output_logits(layer, ins, outs):
        nonlocal no_speech_prob, chunk_logprobs, segment_tokens, chunk_tokens, embedding_weights, has_started
        
        if embedding_weights is None:
            embedding_weights = torch.transpose(model.decoder.token_embedding.weight, 0, 1).to(outs[0].dtype)

        # Get the probability of silence
        if sot_index is not None:
            logits = (outs[0][sot_index,:] @ embedding_weights).float()
            logits = logits.softmax(dim=-1)
            no_speech_prob = logits[tokenizer.no_speech].item()
        
        # Get the log-probabilities of tokens (we don't know yet which one will be chosen)
        if has_started:
            logits = (outs[0][-1:,:] @ embedding_weights).float()
            tokens = torch.cat(chunk_tokens).unsqueeze(0)
            for logit_filter in logit_filters:
                logit_filter.apply(logits, tokens)
            logits = F.log_softmax(logits.squeeze(0), dim=-1)
            chunk_logprobs.append(logits)

    try:

        # Add hooks to the model, to get tokens and attention weights on the fly
        all_hooks = []
        all_hooks.append(model.encoder.conv1.register_forward_hook(hook_mfcc))
        all_hooks.append(model.decoder.token_embedding.register_forward_hook(hook_input_tokens))
        for i, block in enumerate(model.decoder.blocks):
            all_hooks.append(
                block.cross_attn.register_forward_hook(
                    lambda layer, ins, outs, index=i: hook_attention_weights(layer, ins, outs, index))
            )
        if compute_word_confidence or no_speech_threshold is not None:
            all_hooks.append(model.decoder.ln.register_forward_hook(hook_output_logits))

        transcription = model.transcribe(audio,
                                     language=language,
                                     task=task,
                                     fp16=fp16,
                                     temperature=temperature,
                                     best_of=best_of,
                                     beam_size=beam_size,
                                     patience=patience,
                                     length_penalty=length_penalty,
                                     no_speech_threshold=no_speech_threshold,
                                     logprob_threshold=logprob_threshold,
                                     compression_ratio_threshold=compression_ratio_threshold,
                                     condition_on_previous_text=condition_on_previous_text,
                                     initial_prompt=initial_prompt,
                                     suppress_tokens=suppress_tokens,
                                     verbose=verbose
                                     )
    finally:

        # Remove hooks
        for hook in all_hooks:
            hook.remove()

    # Finalize (collect last segment)
    may_flush_segment()
    segment_tokens.pop(-1)

    token_special_idx = min(token_sot, token_eot)

    def filter_tokens(tokens):
        while len(tokens) and tokens[0] >= token_special_idx:
            tokens = tokens[1:]
        while len(tokens) and tokens[-1] >= token_special_idx:
            tokens = tokens[:-1]
        return tokens

    assert len(segment_tokens) == len(timestamped_word_segments), f"Inconsistent number of segments: tokens ({len(segment_tokens)}) != timestamped_word_segments ({len(timestamped_word_segments)})"
    assert len(segment_avglogprobs) == len(segment_tokens), f"Inconsistent number of segments: avg logprobs ({len(segment_avglogprobs)}) != tokens ({len(segment_tokens)})"
    assert len(segment_logprobs) == len(segment_tokens), f"Inconsistent number of segments: logprobs ({len(segment_logprobs)}) != tokens ({len(segment_tokens)})"

    whisper_segments = transcription["segments"]
    l1 = len(whisper_segments)
    l2 = len(timestamped_word_segments)
    if l1 != l2 and l1 != 0:
        logger.warning(f"Inconsistent number of segments: whisper_segments ({l1}) != timestamped_word_segments ({l2})")
    assert l1 == l2 or l1 == 0, f"Inconsistent number of segments: whisper_segments ({l1}) != timestamped_word_segments ({l2})"

    logger.debug("Compile results")
    words = []
    for i, (segment, timestamped_words, token, avglogprob, logprobs) in enumerate(zip(whisper_segments, timestamped_word_segments, segment_tokens, segment_avglogprobs, segment_logprobs)):
        timestamped_tokens = filter_tokens(token)
        whisper_tokens = filter_tokens(segment["tokens"])
        if timestamped_tokens != whisper_tokens:
            assert len(timestamped_tokens) < len(whisper_tokens) and timestamped_tokens == whisper_tokens[:len(timestamped_tokens)], f"Fatal Error: Got inconsistent text for segment {i}:\n{tokenizer.decode(timestamped_tokens)}\n!=\n{tokenizer.decode(whisper_tokens)}"      
            logger.warning(f"Text had to be shortned on segment {i}:\n{tokenizer.decode(timestamped_tokens)}\n!=\n{tokenizer.decode(whisper_tokens)}")

        offset = segment["seek"] * HOP_LENGTH / SAMPLE_RATE
        for timestamped_word in timestamped_words:
            timestamped_word["start"] += offset
            timestamped_word["end"] += offset
            timestamped_word["idx_segment"] = i

        if compute_word_confidence:
            # TODO: make this assert a warning?
            if "avg_logprob_reliable" not in timestamped_words[-1] or timestamped_words[-1]["avg_logprob_reliable"]:
                assert abs(segment["avg_logprob"] - avglogprob) < 1e-2, f"Fatal Error: Got inconsistent logprob for segment {i}: {avglogprob} != {segment['avg_logprob']}"
            if include_punctuation_in_confidence:
                segment["confidence"] = round_confidence(logprobs.mean().exp().item())
            else:
                logprobs_nopunc = []
            i_end = 0
            for timestamped_word in timestamped_words:
                i_start = i_end
                tokens = timestamped_word["tokens"]
                i_end += len(tokens)

                assert i_end <= len(logprobs), f"Fatal Error: Got out-of-bound index for segment {i}: {i_end} > {len(logprobs)}"
                if include_punctuation_in_confidence:
                    word_logprobs = logprobs[i_start:i_end]
                else:
                    tokens_str = [tokenizer.decode([t]) for t in tokens]
                    while len(tokens_str) > 1 and tokens_str[-1][-1] in _punctuation: # Note: look at the last character of token, to take into account "...", "!!", etc.
                        tokens_str = tokens_str[:-1]
                        tokens = tokens[:-1]
                    word_logprobs = logprobs[i_start:i_start + len(tokens)]
                    logprobs_nopunc.append(word_logprobs)

                timestamped_word["confidence"] = round_confidence(word_logprobs.mean().exp().item())

            assert i_end == len(logprobs), f"Fatal Error: Got inconsistent logprobs length for segment {i}: {len(logprobs)} != {i_end}"
            if not include_punctuation_in_confidence:
                logprobs_nopunc = torch.cat(logprobs_nopunc)
                segment["confidence"] = round_confidence(logprobs_nopunc.mean().exp().item())

        if len(timestamped_words):
            segment_start = segment["start"]
            segment_end = segment["end"]

            if timestamped_words[0]["start"] < segment_start - refine_whisper_precision:
                logger.warning(f"Problem on start position for segment {i} ({segment['text']}) : {timestamped_words[0]['start']} << {segment_start}")
            if timestamped_words[-1]["end"] > segment_end + refine_whisper_precision:
                logger.warning(f"Problem on end position for segment {i} ({segment['text']}) : {timestamped_words[0]['end']} >> {segment_end}")
            # assert timestamped_words[0]["start"] >= segment_start - refine_whisper_precision
            # assert timestamped_words[-1]["end"] <= segment_end + refine_whisper_precision

        words.extend(timestamped_words)

    ensure_increasing_positions(words, min_duration=min_word_duration)

    if verbose:
        print(f"Detected {len(words)} words:")
    for word in words:
        if verbose:
            print(f"[{format_timestamp(word['start'])} --> {format_timestamp(word['end'])}]  {word['text']}")
        word.pop("tokens")
        if "avg_logprob_reliable" in word:
            word.pop("avg_logprob_reliable")
        idx_segment = word.pop("idx_segment")
        segment = whisper_segments[idx_segment]
        if "words" in segment:
            segment["words"].append(word)
        else:
            segment["words"] = [word]
            if refine_whisper_precision:
                segment["start"] = word["start"]
        if refine_whisper_precision:
            segment["end"] = word["end"]

    return transcription

def perform_word_alignment(
    tokens, attention_weights,
    tokenizer,
    use_space=True,
    refine_whisper_precision_nsamples=0,
    medfilt_width=9,
    qk_scale=1.0,
    most_top_layers=None,  # 6
    mfcc=None,
    plot=False,
    remove_punctuation_from_words=False,
    unfinished_decoding=False,
    debug=False,
):
    """
    Perform word alignment on the given tokens and attention weights.
    Returns a list of (word, start_time, end_time) tuples.

    tokens: list of tokens (integers)
    attention_weights: list of attention weights (torch tensors)
    tokenizer: tokenizer used to tokenize the text
    use_space: whether to use spaces to split the tokens into words (should be true for all languages except Japanese, Chinese, ...)
    refine_whisper_precision_nsamples: precision time
    """

    assert len(tokens) > 1, f"Got unexpected sequence of tokens of length {len(tokens)} {tokenizer.decode_with_timestamps(tokens)}"
    start_token = tokens[0] - tokenizer.timestamp_begin
    end_token = tokens[-1] - tokenizer.timestamp_begin

    # Check start / end tokens
    if start_token < 0:
        raise RuntimeError(f"Missing start token in {tokenizer.decode_with_timestamps(tokens)}")
    if len(tokens) == 1 or end_token < 0:
        # This can happens when Whisper is stucked as a Language Model
        if debug:
            logger.debug(f"Missing end token in {tokenizer.decode_with_timestamps(tokens)}")
        end_token = N_FRAMES // 2
    if end_token == start_token and refine_whisper_precision_nsamples == 0:
        if debug:
            logger.debug(f"Got empty segment in {tokenizer.decode_with_timestamps(tokens)}")
        return []

    # Put some margin around the segment
    if refine_whisper_precision_nsamples > 0:
        start_token = max(start_token - refine_whisper_precision_nsamples, 0)
        end_token = min(end_token + refine_whisper_precision_nsamples, N_FRAMES // 2)

    # Get the limit of audio duration
    max_duration = None
    if mfcc is not None:
        max_duration = find_start_padding(mfcc)
        if max_duration is not None:
            max_duration = max_duration // 2

    if end_token <= start_token:
        raise RuntimeError(f"Got segment with null or negative duration {tokenizer.decode_with_timestamps(tokens)}: {start_token} {end_token}")

    start_time = start_token * AUDIO_TIME_PER_TOKEN
    end_time = end_token * AUDIO_TIME_PER_TOKEN

    split_tokens = split_tokens_on_spaces if use_space else split_tokens_on_unicode
    words, word_tokens = split_tokens(tokens, tokenizer, remove_punctuation_from_words=remove_punctuation_from_words)

    for i, w in enumerate(attention_weights):
        assert w.shape[-2] == len(tokens), f"Attention weights have wrong shape: {w.shape[-2]} (expected {len(tokens)})."
    weights = torch.cat(attention_weights) # layers * heads * tokens * frames

    num_tokens = weights.shape[-2]
    num_frames = end_token - start_token
    if num_tokens > num_frames:
        # TODO: test this case
        logger.warning(f"Too much text ({num_tokens} tokens) for the given number of frames ({num_frames}) in: {tokenizer.decode_with_timestamps(tokens)}\nThe end of the text will be removed.")
        return perform_word_alignment(
            tokens[:num_frames-1] + [tokens[-1]],
            [[w[:, :, :num_frames-1, :], w[:, :, -1:, :]]
                for w in attention_weights],
            tokenizer,
            use_space=use_space,
            refine_whisper_precision_nsamples=refine_whisper_precision_nsamples,
            medfilt_width=medfilt_width,
            qk_scale=qk_scale,
            most_top_layers=most_top_layers,
            mfcc=mfcc,
            plot=plot,
            remove_punctuation_from_words=remove_punctuation_from_words,
            unfinished_decoding=True,
            debug=debug,
        )

    assert end_token <= weights.shape[-1]
    assert len(tokens) == num_tokens

    weights = weights[:, :, :, start_token: end_token].cpu()

    weights = median_filter(weights, (1, 1, 1, medfilt_width))

    weights = torch.tensor(weights * qk_scale).softmax(dim=-1)
    # weights = weights.softmax(dim=-2)
    # TODO: Do we really need this?
    weights = weights / weights.norm(dim=-2, keepdim=True)

    if most_top_layers:
        weights = weights[-most_top_layers:]  # at most 6 top layers
    weights = weights.mean(axis=(0, 1))  # average over layers and heads
    weights = -weights.double().numpy()

    # Enforce the max duration
    if max_duration:
        if start_token >= max_duration:
            logger.warn(f"Got start time outside of audio boundary")
        else:
            weights[:-1,max_duration:] = 0

    # Similar as "symmetric1" but without the possibility to have several timestamps for two tokens
    step_pattern = dtw.stepPattern.StepPattern(dtw.stepPattern._c(
        1, 1, 1, -1,
        1, 0, 0, 1,
        2, 0, 1, -1,
        2, 0, 0, 1,
    ))
    alignment = dtw.dtw(weights, step_pattern=step_pattern)

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        word_tokens_str = [[tokenizer.decode_with_timestamps([ti]) for ti in t] for t in word_tokens]

        if mfcc is None:
            plt.figure(figsize=(16, 9), frameon=False)
        else:
            plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [3, 1]})
            plt.subplot(2, 1, 1, frameon=False)

        plt.imshow(-weights, aspect="auto")
        plt.plot(alignment.index2s, alignment.index1s, color="red")

        xticks = np.arange(0, weights.shape[1], 1 / AUDIO_TIME_PER_TOKEN)
        xticklabels = [round_timestamp(x) for x in xticks * AUDIO_TIME_PER_TOKEN + start_time]

        ylims = plt.gca().get_ylim()

        ax = plt.gca()
        ax.tick_params('both', length=0, width=0, which='minor', pad=6)

        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_label_position("left")
        ax.invert_yaxis()
        ax.set_ylim(ylims)

        major_ticks = [-0.5]
        minor_ticks = []
        current_y = 0

        for word, word_token in zip(words, word_tokens):
            minor_ticks.append(current_y + len(word_token) / 2 - 0.5)
            current_y += len(word_token)
            major_ticks.append(current_y - 0.5)

        words_with_subwords = ["|".join(s) for (w, s) in zip(words, word_tokens_str)]

        ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
        ax.yaxis.set_minor_formatter(
            ticker.FixedFormatter(words_with_subwords))
        ax.set_yticks(major_ticks)
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        for y in major_ticks:
            plt.axhline(y, color="black", linestyle="dashed")

        plt.ylabel("Words")

        if mfcc is not None:
            plt.xticks(xticks)
            plt.setp(plt.gca().get_xticklabels(), visible=False)

            xticks *= 2

            plt.subplot(2, 1, 2, frameon=False)
            plt.imshow(mfcc[0, :, start_token * 2: end_token * 2].cpu(), aspect="auto")
            plt.yticks([])
            plt.ylabel("MFCC")

        plt.xticks(xticks, xticklabels)
        plt.xlabel("Time (s)")

    jumps = np.diff(alignment.index1s)
    jumps = np.pad(jumps, (1, 0), constant_values=1)
    jumps = jumps.astype(bool)
    jumps = alignment.index2s[jumps]
    jump_times = jumps * AUDIO_TIME_PER_TOKEN
    jump_times = np.pad(jump_times, (0, 1),
                        constant_values=end_time - start_time)

    # display the word-level timestamps in a table
    word_boundaries = np.cumsum([len(t) for t in word_tokens])
    word_boundaries = np.pad(word_boundaries, (1, 0))
    begin_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]

    # Ignore start / end tokens
    if not refine_whisper_precision_nsamples:
        begin_times[1] = begin_times[0]
    if not refine_whisper_precision_nsamples:
        end_times[-2] = end_times[-1]
    if unfinished_decoding:
        words = words[1:]
        word_tokens = word_tokens[1:]
        begin_times = begin_times[1:]
        end_times = end_times[1:]
    else:
        words = words[1:-1]
        word_tokens = word_tokens[1:-1]
        begin_times = begin_times[1:-1]
        end_times = end_times[1:-1]

    if plot:
        ymin = 1

        if mfcc is not None:
            for i, (begin, end) in enumerate(zip(begin_times, end_times)):
                for x in [begin, end,] if i == 0 else [end,]:
                    plt.axvline(x * 2 / AUDIO_TIME_PER_TOKEN,
                                color="red", linestyle="dotted")

            plt.subplot(2, 1, 1)

        for i, (w, ws, begin, end) in enumerate(zip(words, word_tokens, begin_times, end_times)):
            ymax = ymin + len(ws)
            plt.text(begin / AUDIO_TIME_PER_TOKEN, num_tokens,
                     w, ha="left", va="top", color="red")
            for x in [begin, end,] if i == 0 else [end,]:
                plt.axvline(x / AUDIO_TIME_PER_TOKEN, color="red", linestyle="dotted",
                            ymin=1-ymin/num_tokens,
                            ymax=0,  # 1-ymax/num_tokens,
                            )
            ymin = ymax

        plt.show()

    return [
        dict(
            text=word,
            start=round_timestamp(begin + start_time),
            end=round_timestamp(end + start_time),
            tokens=tokens,
        )
        for word, begin, end, tokens in zip(words, begin_times, end_times, word_tokens)
        if not word.startswith("<|")
    ]

def find_start_padding(mfcc):
    """ Return start of padding given the mfcc, or None if there is no padding """
    last_mfcc = mfcc[0, :, -1]
    if torch.min(last_mfcc) == torch.max(last_mfcc) == 0:
        candidate_index = mfcc.shape[-1] - 2
        while candidate_index > 0:
            candidate = mfcc[0, :, candidate_index]
            if not torch.equal(candidate, last_mfcc):
                return candidate_index + 1
            candidate_index -= 1
        return 0 # WTF!?

def round_confidence(x):
    return round(x, 3)

def round_timestamp(x):
    return round(x, 2)

_punctuation = "".join(c for c in string.punctuation if c not in ["-", "'"])

def split_tokens_on_unicode(tokens: list, tokenizer, tokens_as_string=False, remove_punctuation_from_words=False, isolate_punctuations=False):
    words = []
    word_tokens = []
    current_tokens = []

    for token in tokens:
        current_tokens.append(token)
        decoded = tokenizer.decode_with_timestamps(current_tokens)
        if "\ufffd" not in decoded:
            punctuation = not isolate_punctuations and (decoded.strip() and decoded.strip() in _punctuation)
            if punctuation:
                if len(words) == 0:
                    words = [""]
                    word_tokens = [[]]
                if not remove_punctuation_from_words:
                    words[-1] += decoded
                if tokens_as_string:
                    word_tokens[-1].append(decoded.strip())
                else:
                    word_tokens[-1].extend(current_tokens)
            else:
                words.append(decoded)
                word_tokens.append(
                    [decoded.strip()] if tokens_as_string else current_tokens)
            current_tokens = []

    return words, word_tokens


def split_tokens_on_spaces(tokens: torch.Tensor, tokenizer, tokens_as_string=False, remove_punctuation_from_words=False):
    subwords, subword_tokens_list = split_tokens_on_unicode(
        tokens, tokenizer, tokens_as_string=tokens_as_string, remove_punctuation_from_words=remove_punctuation_from_words)
    words = []
    word_tokens = []

    for i, (subword, subword_tokens) in enumerate(zip(subwords, subword_tokens_list)):
        special = (subword_tokens[0].startswith("<|")) if tokens_as_string else (subword_tokens[0] >= tokenizer.eot)
        previous_special = i > 0 and (subword_tokens_list[i-1][0].startswith("<|")) if tokens_as_string else (subword_tokens_list[i-1][0] >= tokenizer.eot)
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in _punctuation
        if special or (with_space and not punctuation) or previous_special:
            words.append(subword.strip())
            word_tokens.append(subword_tokens)
        else:
            words[-1] = words[-1] + subword.strip()
            word_tokens[-1].extend(subword_tokens)

    return words, word_tokens

def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def ensure_increasing_positions(segments, min_duration=0):
    """
    Ensure that "start" and "end" come in increasing order
    """
    has_modified_backward = False
    previous_end = 0
    for i, seg in enumerate(segments):
        if seg["start"] < previous_end:
            assert i > 0
            new_start = round_timestamp((previous_end + seg["start"]) / 2)
            if new_start < segments[i-1]["start"] + min_duration:
                new_start = previous_end
            else:
                segments[i-1]["end"] = new_start
                has_modified_backward = True
            seg["start"] = new_start
        if seg["end"] <= seg["start"] + min_duration:
            seg["end"] = seg["start"] + min_duration
        previous_end = seg["end"]
    if has_modified_backward:
        return ensure_increasing_positions(segments, min_duration)

    previous_end = 0
    for seg in segments:
        seg["start"] = round_timestamp(seg["start"])
        seg["end"] = round_timestamp(seg["end"])
        assert seg["start"] >= previous_end, f"Got segment {seg} coming before the previous finishes ({previous_end})"
        assert seg["end"] > seg["start"], f"Got segment {seg} with end <= start"
        previous_end = seg["end"]

    return segments

## Some utilities for writing transcripts to files

# def remove_punctuation(text):
#     return text.translate(str.maketrans('', '', _punctuations))

def write_vtt_words(transcript, file):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        for word in segment["words"]:
            print(
                f"{format_timestamp(word['start'])} --> {format_timestamp(word['end'])}\n"
                f"{word['text']}\n",
                file=file,
                flush=True,
            )

def write_srt_words(transcript, file):
    i = 1
    for segment in transcript:
        for word in segment["words"]:
            print(
                f"{i}\n"
                f"{format_timestamp(word['start'], always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(word['end'], always_include_hours=True, decimal_marker=',')}\n"
                f"{word['text']}\n",
                file=file,
                flush=True,
            )
            i += 1

def write_csv(transcript, file):
    # Use csv to write
    csv.writer(file).writerows(
        [[segment["text"].strip(), segment["start"], segment["end"]] for segment in transcript]
    )

def write_csv_words(transcript, file):
    writer = csv.writer(file)
    for segment in transcript:
        for word in segment["words"]:
            writer.writerow([word['text'], word['start'], word['end']])

def cli():

    import os
    import sys
    import argparse
    import json

    from whisper.utils import str2bool, optional_float, optional_int, write_txt, write_srt, write_vtt

    parser = argparse.ArgumentParser(
        description='Transcribe a single audio with whisper and compute word timestamps',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('audio', help="Audio file to transcribe", nargs='+')
    parser.add_argument('--model', help=f"Name of the Whisper model to use.", choices=whisper.available_models(), default="small")
    parser.add_argument("--model_dir", default=None, help="The path to save model files; uses ~/.cache/whisper by default", type=str)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", default=None, help="directory to save the outputs", type=str)
    parser.add_argument('--plot', help="Plot word alignments", default=False, action="store_true")
    parser.add_argument("--verbose", type=str2bool, default=False, help="Whether to print out the progress and debug messages of Whisper")

    parser.add_argument("--punctuations", default=True, help="Whether to include punctuations within the words", type=str2bool)
    parser.add_argument("--txt", default=True, help="Whether to save in simple text format", type=str2bool)
    parser.add_argument("--vtt", default=True, help="Whether to save in VTT format", type=str2bool)
    parser.add_argument("--srt", default=True, help="Whether to save in SRT format", type=str2bool)
    parser.add_argument("--csv", default=False, help="Whether to save in CSV format", type=str2bool)
    parser.add_argument("--json", default=False, help="Whether to save in JSON format", type=str2bool)
    
    parser.add_argument("--task", default="transcribe", help="Whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')", choices=["transcribe", "translate"], type=str)
    parser.add_argument('--language', help=f"Language to use. Among : {', '.join(sorted(k+'('+v+')' for k,v in whisper.tokenizer.LANGUAGES.items()))}.", choices=sorted(whisper.tokenizer.LANGUAGES.keys()) + sorted([k.title() for k in whisper.tokenizer.TO_LANGUAGE_CODE.keys()]), default=None)
    
    parser.add_argument("--temperature", default=0.0, help="Temperature to use for sampling", type=float)
    # TODO: implement default best_of 5
    parser.add_argument("--best_of", type=optional_int, default=None, help="number of candidates when sampling with non-zero temperature")
    # TODO: implement default beam_size 5
    parser.add_argument("--beam_size", type=optional_int, default=None, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations", type=str)
    parser.add_argument("--initial_prompt", default=None, help="optional text to provide as a prompt for the first window.", type=str)
    parser.add_argument("--condition_on_previous_text", default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop", type=str2bool)
    parser.add_argument("--fp16", default=None, help="whether to perform inference in fp16; Automatic by default (True if GPU available, False otherwise)", type=str2bool)

    # TODO: implement default support 0.2
    parser.add_argument("--temperature_increment_on_fallback", default=None, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below", type=optional_float)
    parser.add_argument("--compression_ratio_threshold", default=2.4, help="If the gzip compression ratio is higher than this value, treat the decoding as failed", type=optional_float)
    parser.add_argument("--logprob_threshold", default=-1.0, help="If the average log probability is lower than this value, treat the decoding as failed", type=optional_float)
    parser.add_argument("--no_speech_threshold", default=0.6, help="If the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence", type=optional_float)
    parser.add_argument("--threads", default=0, help="Number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS", type=optional_int)

    parser.add_argument('--debug', help="Print some debug information for word alignement", default=False, action="store_true")

    args = parser.parse_args().__dict__

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    threads = args.pop("threads")
    if threads:
        torch.set_num_threads(threads)

    audio_files = args.pop("audio")
    
    model = args.pop("model")
    device = args.pop("device")
    model_dir = args.pop("model_dir")

    csv_out = args.pop("csv")
    json_out = args.pop("json")
    srt_out = args.pop("srt")
    txt_out = args.pop("txt")
    vtt_out = args.pop("vtt")



    model = whisper.load_model(model, device=device, download_root=model_dir)

    plot_word_alignment = args.pop("plot")

    debug = args.pop("debug")
    if debug:
        logger.setLevel(logging.DEBUG)
        # This supposes to plug a logger with name "WHISPER" into Whisper source code (no harm if it's not set)
        logging.getLogger("WHISPER").setLevel(logging.DEBUG)

    output_dir = args.pop("output_dir")
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for audio_path in audio_files:

        result = transcribe_timestamped(
            model, audio_path,
            temperature=temperature,
            plot_word_alignment=plot_word_alignment,
            remove_punctuation_from_words=not args.pop("punctuations"),
            **args
        )

        if output_dir:

            outname = os.path.join(output_dir, os.path.basename(audio_path))
            if json_out:
                # save JSON
                with open(outname + ".words.json", "w", encoding="utf-8") as js:
                    json.dump(result, js, indent=2, ensure_ascii=False)

            # save TXT
            if txt_out:
                with open(outname + ".txt", "w", encoding="utf-8") as txt:
                    write_txt(result["segments"], file=txt)

            # save VTT
            if vtt_out:
                with open(outname + ".vtt", "w", encoding="utf-8") as vtt:
                    write_vtt(result["segments"], file=vtt)
                with open(outname + ".words.vtt", "w", encoding="utf-8") as vtt:
                    write_vtt_words(result["segments"], file=vtt)

            # save SRT
            if srt_out:
                with open(outname + ".srt", "w", encoding="utf-8") as srt:
                    write_srt(result["segments"], file=srt)
                with open(outname + ".words.srt", "w", encoding="utf-8") as srt:
                    write_srt_words(result["segments"], file=srt)

            # save CSV
            if csv_out:
                with open(outname + ".csv", "w", encoding="utf-8") as csv:
                    write_csv(result["segments"], file=csv)
                with open(outname + ".words.csv", "w", encoding="utf-8") as csv:
                    write_csv_words(result["segments"], file=csv)

        elif not args["verbose"]:

            json.dump(result, sys.stdout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    cli()