#!/usr/bin/env python3

__author__ = "Jérôme Louradour"
__credits__ = ["Jérôme Louradour"]
__license__ = "MIT"

# Whisper
import whisper
import torch

# For alignment
import numpy as np
import dtw
import scipy.signal

# Additional for text tokenization
import string

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
    refine_whisper_precision=0.5,
    min_word_duration=0.1,
    plot_word_alignment=False,
    # other Whisper options
    temperature=0.0, # TODO: support list
    best_of=5,
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

    refine_whisper_precision: float
        How much can we refine Whisper segment positions, in seconds. Must be a multiple of 0.02.

    min_word_duration: float
        Minimum duration of a word, in seconds. If a word is shorter than this, timestamps will be adjusted.

    plot_word_alignment: bool
        Whether to plot the word alignment for each segment. matplotlib must be installed to use this option.

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

    assert refine_whisper_precision >= 0 and refine_whisper_precision / AUDIO_TIME_PER_TOKEN == round(
        refine_whisper_precision / AUDIO_TIME_PER_TOKEN), f"refine_whisper_precision must be a positive multiple of {AUDIO_TIME_PER_TOKEN}"
    refine_whisper_precision_nsamples = round(
        refine_whisper_precision / AUDIO_TIME_PER_TOKEN)

    if isinstance(temperature, (list, tuple)) and len(temperature) > 1:
        raise NotImplementedError("Transcription with several temperatures not implemented")
    if beam_size is not None:
        raise NotImplementedError("Transcription with beam search not implemented")

    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language=language)
    # Note: we cannot trust the token in the middle of tokenizer.sot_sequence which refers to the language
    #       (arbitrarily set to <|en|> if it's actually None/unknown)
    token_sot = tokenizer.sot_sequence[0]
    token_transcribe = tokenizer.sot_sequence[-1]
    token_eot = tokenizer.eot

    input_stride = N_FRAMES // model.dims.n_audio_ctx
    time_precision = input_stride * HOP_LENGTH / SAMPLE_RATE
    assert time_precision == AUDIO_TIME_PER_TOKEN

    use_space = whisper.tokenizer.TO_LANGUAGE_CODE.get(str(language).lower(), language) not in ["zh", "ja", "th", "lo", "my"]

    # install hooks on the cross attention layers to retrieve the attention weights and corresponding tokens
    tokens = [[]]
    timestamped_word_segments = []
    attention_weights = [[] for _ in range(model.dims.n_text_layer)]
    mfcc = None  # For plotting only

    num_inference_steps = 0  # For debug only

    def get_attention_weights(layer, ins, outs, index):
        nonlocal attention_weights
        # On old version of whisper output is a single tensor
        assert isinstance(outs, tuple) and len(outs) == 2
        attention_weights[index].append(outs[-1])

    def reset_new_segment(timestamp_start, add_even_if_missing_end_token=True):
        nonlocal tokens, attention_weights
        nonlocal tokenizer, token_transcribe

        if timestamp_start is None:
            tokens.append([])
        else:
            tokens[-1] = tokens[-1][:-1]
            tokens.append([timestamp_start])
            
        current_tokens = tokens[-2]
        current_attention_weights = [torch.cat(w, dim=-2)[:, :, :-1, :] for w in attention_weights]

        if debug:
            logger.debug(f"add new segment {len(tokens)-1}: mode {1 if timestamp_start is None else 0}: {tokenizer.decode_with_timestamps(current_tokens)}")
        if token_transcribe in current_tokens:
            i_transcribe = current_tokens.index(token_transcribe)
            current_tokens = current_tokens[i_transcribe+1:]
            current_attention_weights = [w[:, :, i_transcribe+1:, :] for w in current_attention_weights]
            tokens[-2] = current_tokens
            if debug:
                logger.debug(f"refine segment {len(tokens)-1} to: {tokenizer.decode_with_timestamps(current_tokens)}")

        ws = perform_word_alignment(
            current_tokens,
            current_attention_weights,
            tokenizer,
            use_space=use_space,
            refine_whisper_precision_nsamples=refine_whisper_precision_nsamples,
            add_even_if_missing_end_token=add_even_if_missing_end_token,
            mfcc=mfcc,
            plot=plot_word_alignment,
        )
        if len(ws):
            timestamped_word_segments.append(ws)
        else:
            if debug:
                logger.debug(f"Not adding segment ({len(timestamped_word_segments)}) {tokenizer.decode_with_timestamps(tokens[-2])}")
            tokens.pop(-2)

        attention_weights = [[w[-1][:, :, -1:, :]] for w in attention_weights]

    def get_input_tokens(layer, ins, outs):
        nonlocal tokens, num_inference_steps, attention_weights
        curr_tokens = ins[0][0]
        num_inference_steps += 1

        if len(curr_tokens) > 5:
            ends_with_sot = (curr_tokens[-1] == token_transcribe and curr_tokens[-3] == token_sot)
            if ends_with_sot and curr_tokens[-4] >= tokenizer.timestamp_begin:
                reset_new_segment(None)

        last_token = tokens[-1][-1] if len(tokens[-1]) > 0 else -1
        tokens[-1] += curr_tokens.tolist()

        is_a_start = curr_tokens[-1] == tokenizer.timestamp_begin
        is_a_timestamp = (len(curr_tokens) ==
                          1 and curr_tokens[0] >= tokenizer.timestamp_begin)
        is_last_timestamp = last_token > tokenizer.timestamp_begin

        if is_a_start:

            curr_tokens = tokens[-1]
            ends_with_sot = len(curr_tokens) > 5 and (curr_tokens[-2] == token_transcribe and curr_tokens[-4] == token_sot)

            # Flush
            if len(tokens) > 1 and curr_tokens[0] == tokenizer.sot_prev:
                previous_tokens = tokens[-2]
                if not ends_with_sot or len(previous_tokens) <= 1 or not (len(curr_tokens) >= len(previous_tokens)+3 and curr_tokens[-len(previous_tokens)-4:-4] == previous_tokens):
                    if debug:
                        logger.debug(f"flushing (-1): {tokenizer.decode_with_timestamps(previous_tokens)}")
                    tokens.pop(-2)
                    timestamped_word_segments.pop(-1)
                elif debug:
                    logger.debug(f"NOT flushing: {tokenizer.decode_with_timestamps(previous_tokens)}")
            if ends_with_sot and curr_tokens[-5] < tokenizer.timestamp_begin:
                # Find the last index of <|0.0|> before the last one
                i_start = len(curr_tokens) - 2 - curr_tokens[-2::-1].index(tokenizer.timestamp_begin)
                # Add the missing ending timestamp (set it to the maximum)
                if debug:
                    logger.debug(f"flushing (1): {tokenizer.decode_with_timestamps(curr_tokens[:i_start])}")
                tokens[-1] = curr_tokens[i_start:-4] + ([tokenizer.timestamp_begin + N_FRAMES / 2] * 2)
                attention_weights = [[torch.cat(w, dim=-2)[:, :, i_start:-2, :]] for w in attention_weights]
                reset_new_segment(tokenizer.timestamp_begin)
            else:
                if debug:
                    logger.debug(f"flushing (2): {tokenizer.decode_with_timestamps(curr_tokens[:-1])}")
                tokens[-1] = [curr_tokens[-1]]
                attention_weights = [[w[-1][:, :, -1:, :]] for w in attention_weights]

        elif is_a_timestamp and is_last_timestamp:

            timestamp_token = curr_tokens[0].item()

            reset_new_segment(timestamp_token)

        # elif is_last_timestamp and not is_a_timestamp:
        #         pass

    if plot_word_alignment:
        def get_mfcc(layer, ins, outs):
            nonlocal mfcc
            mfcc = ins[0]
        model.encoder.conv1.register_forward_hook(get_mfcc)

    model.decoder.token_embedding.register_forward_hook(
        lambda layer, ins, outs: get_input_tokens(layer, ins, outs))
    for i, block in enumerate(model.decoder.blocks):
        block.cross_attn.register_forward_hook(
            lambda layer, ins, outs, index=i: get_attention_weights(layer, ins, outs, index))

    if fp16 is None:
        fp16 = model.device != torch.device("cpu")

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
                                     verbose=verbose,
                                     )

    # Finalize
    reset_new_segment(None, False)
    tokens = tokens[:-1]

    token_special_idx = min(token_sot, token_eot)

    def filter_tokens(tokens):
        while len(tokens) and tokens[0] >= token_special_idx:
            tokens = tokens[1:]
        while len(tokens) and tokens[-1] >= token_special_idx:
            tokens = tokens[:-1]
        return tokens

    assert len(tokens) == len(timestamped_word_segments), f"Inconsistent number of segments: tokens ({len(tokens)}) != timestamped_word_segments ({len(timestamped_word_segments)})"

    whisper_segments = transcription["segments"]
    l1 = len(whisper_segments)
    l2 = len(timestamped_word_segments)
    if l1 != l2 and l1 != 0:
        logger.warning(f"Inconsistent number of segments: whisper_segments ({l1}) != timestamped_word_segments ({l2})")
    assert l1 == l2 or l1 == 0, f"Inconsistent number of segments: whisper_segments ({l1}) != timestamped_word_segments ({l2})"

    words = []
    for i, (segment, timestamped_words, token) in enumerate(zip(whisper_segments, timestamped_word_segments, tokens)):
        timestamped_tokens = filter_tokens(token)
        whisper_tokens = filter_tokens(segment["tokens"])
        if timestamped_tokens != whisper_tokens:
            logger.warning(f"Got inconsistent segments at index {i}:\n{tokenizer.decode(timestamped_tokens)}\n!=\n{tokenizer.decode(whisper_tokens)}")
            assert len(timestamped_tokens) < len(whisper_tokens) and timestamped_tokens == whisper_tokens[:len(timestamped_tokens)], f"Got inconsistent segments at index {i}:\n{tokenizer.decode(timestamped_tokens)}\n!=\n{tokenizer.decode(whisper_tokens)}"

        offset = segment["seek"] * HOP_LENGTH / SAMPLE_RATE
        for timestamped_word in timestamped_words:
            timestamped_word["start"] += offset
            timestamped_word["end"] += offset
            timestamped_word["idx_segment"] = i

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
        idx_segment = word.pop("idx_segment")
        segment = whisper_segments[idx_segment]
        if "words" in segment:
            segment["words"].append(word)
        else:
            segment["words"] = [word]
            segment["start"] = word["start"]
        segment["end"] = word["end"]

    return transcription

def perform_word_alignment(
    tokens, attention_weights,
    tokenizer,
    use_space=True,
    refine_whisper_precision_nsamples=0,
    add_even_if_missing_end_token=True,
    medfilt_width=9,
    qk_scale=1.0,
    most_top_layers=None,  # 6
    mfcc=None,
    plot=False,
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

    for i, w in enumerate(attention_weights):
        assert w.shape[-2] == len(tokens), f"Attention weights have wrong shape: {w.shape[-2]} (expected {len(tokens)})."

    assert len(tokens) > 0, f"Got unexpected empty sequence of tokens"
    start_token = tokens[0] - tokenizer.timestamp_begin
    end_token = tokens[-1] - tokenizer.timestamp_begin

    if start_token < 0:
        raise RuntimeError(f"Missing start token in {tokenizer.decode_with_timestamps(tokens)}")
    if len(tokens) == 1 or end_token < 0:
        if add_even_if_missing_end_token:
            if debug:
                logger.debug(f"Missing end token in {tokenizer.decode_with_timestamps(tokens)}")
            return [dict(text="", start=round(start_token * AUDIO_TIME_PER_TOKEN, 2), end=round(end_token * AUDIO_TIME_PER_TOKEN, 2))]
        else:
            return []
    if end_token == start_token and refine_whisper_precision_nsamples == 0:
        if debug:
            logger.debug(f"Got empty segment in {tokenizer.decode_with_timestamps(tokens)}")
        return []

    if refine_whisper_precision_nsamples > 0:
        start_token = max(start_token - refine_whisper_precision_nsamples, 0)
        end_token = min(end_token + refine_whisper_precision_nsamples, N_FRAMES // 2)

    if end_token <= start_token:
        raise RuntimeError(f"Got segment with null or negative duration {tokenizer.decode_with_timestamps(tokens)}: {start_token} {end_token}")

    start_time = start_token * AUDIO_TIME_PER_TOKEN
    end_time = end_token * AUDIO_TIME_PER_TOKEN

    split_tokens = split_tokens_on_spaces if use_space else split_tokens_on_unicode
    words, word_tokens = split_tokens(tokens, tokenizer)

    weights = torch.cat(attention_weights) # layers * heads * tokens * frames

    num_tokens = weights.shape[-2]
    num_frames = end_token - start_token
    if num_tokens > num_frames:
        logger.warning(f"Too many tokens ({num_tokens}) given the number of frames ({num_frames}) in: {tokenizer.decode_with_timestamps(tokens)}")
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
        )

    assert end_token <= weights.shape[-1]
    assert len(tokens) == num_tokens

    weights = weights[:, :, :, start_token: end_token].cpu()

    weights = scipy.signal.medfilt(weights, (1, 1, 1, medfilt_width))

    weights = torch.tensor(weights * qk_scale).softmax(dim=-1)
    # weights = weights.softmax(dim=-2)
    # TODO: Do we really need this?
    weights = weights / weights.norm(dim=-2, keepdim=True)

    if most_top_layers:
        weights = weights[-most_top_layers:]  # at most 6 top layers
    weights = weights.mean(axis=(0, 1))  # average over layers and heads
    weights = -weights.double().numpy()

    # We could enforce to not go outside real boundaries of the segments, for words in the middle...
    # if refine_whisper_precision_start:
    #     weights[1 + len(word_tokens[1]):, :refine_whisper_precision_start] = 0
    #     weights[0, refine_whisper_precision_start*2:] = 0
    # if refine_whisper_precision_end:
    #     weights[:-(1 + len(word_tokens[-2])), -refine_whisper_precision_end:] = 0
    #     weights[-1, :-refine_whisper_precision_end*2] = 0

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

        if mfcc is None:
            plt.figure(figsize=(16, 9), frameon=False)
        else:
            plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={
                         'height_ratios': [3, 1]})
            plt.subplot(2, 1, 1, frameon=False)

        plt.imshow(-weights, aspect="auto")
        plt.plot(alignment.index2s, alignment.index1s, color="red")

        xticks = np.arange(0, weights.shape[1], 1 / AUDIO_TIME_PER_TOKEN)
        xticklabels = [round(x, 2) for x in xticks * AUDIO_TIME_PER_TOKEN + start_time]

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

        words_with_subwords = [
            w if len(s) == 1 else "|".join(s)
            for (w, s) in zip(words, word_tokens)
        ]

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
            plt.imshow(mfcc[0, :, start_token *
                       2: end_token * 2], aspect="auto")
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
    words = words[1:-1]
    begin_times = begin_times[1:-1]
    end_times = end_times[1:-1]

    if plot:
        word_tokens = word_tokens[1:-1]
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
        dict(text=word, start=round(begin + start_time, 2),
             end=round(end + start_time, 2))
        for word, begin, end in zip(words, begin_times, end_times)
        if not word.startswith("<|")
    ]


def split_tokens_on_unicode(tokens: list, tokenizer, tokens_as_string=True):
    words = []
    word_tokens = []
    current_tokens = []

    for token in tokens:
        current_tokens.append(token)
        decoded = tokenizer.decode_with_timestamps(current_tokens)
        if "\ufffd" not in decoded:
            words.append(decoded)
            word_tokens.append(
                [decoded.strip()] if tokens_as_string else current_tokens)
            current_tokens = []

    return words, word_tokens


def split_tokens_on_spaces(tokens: torch.Tensor, tokenizer, tokens_as_string=True):
    subwords, subword_tokens_list = split_tokens_on_unicode(
        tokens, tokenizer, tokens_as_string=tokens_as_string)
    words = []
    word_tokens = []

    for i, (subword, subword_tokens) in enumerate(zip(subwords, subword_tokens_list)):
        special = (subword_tokens[0].startswith("<|")) if tokens_as_string else (subword_tokens[0] >= tokenizer.eot)
        previous_special = i > 0 and (subword_tokens_list[i-1][0].startswith("<|")) if tokens_as_string else (subword_tokens_list[i-1][0] >= tokenizer.eot)
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in string.punctuation
        if special or (with_space and not punctuation) or previous_special:
            words.append(subword.strip())
            word_tokens.append(subword_tokens)
        else:
            words[-1] = words[-1] + subword.strip()
            word_tokens[-1].extend(subword_tokens)

    return words, word_tokens



def ensure_increasing_positions(segments, min_duration=0.1):
    """
    Ensure that "start" and "end" come in increasing order
    """
    has_modified_backward = False
    previous_end = 0
    for i, seg in enumerate(segments):
        if seg["start"] < previous_end:
            assert i > 0
            new_start = round((previous_end + seg["start"]) / 2, 2)
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
        seg["start"] = round(seg["start"], 2)
        seg["end"] = round(seg["end"], 2)
        assert seg["start"] >= previous_end, f"Got segment {seg} coming before the previous finishes ({previous_end})"
        assert seg["end"] > seg["start"], f"Got segment {seg} with end <= start"
        previous_end = seg["end"]

    return segments


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

def write_csv_words(transcript, file):
    for segment in transcript:
        for word in segment["words"]:
            #strip punctuation from the isolated words
            stripChars ='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'
            aWord = word['text'].translate(str.maketrans('','',stripChars))
            print(
                f"{aWord},{word['start']},{word['end']}",
                file=file,
                flush=True,
            )

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

    parser.add_argument("--csv", default=True, help="Whether to save in CSV format", type=str2bool)
    parser.add_argument("--json", default=False, help="Whether to save in JSON format", type=str2bool)
    parser.add_argument("--srt", default=True, help="Whether to save in SRT format", type=str2bool)
    parser.add_argument("--vtt", default=True, help="Whether to save in VTT format", type=str2bool)
    parser.add_argument("--txt", default=True, help="Whether to save in simple text format", type=str2bool)
    
    parser.add_argument("--task", default="transcribe", help="Whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')", choices=["transcribe", "translate"], type=str)
    parser.add_argument('--language', help=f"Language to use. Among : {', '.join(sorted(k+'('+v+')' for k,v in whisper.tokenizer.LANGUAGES.items()))}.", choices=sorted(whisper.tokenizer.LANGUAGES.keys()) + sorted([k.title() for k in whisper.tokenizer.TO_LANGUAGE_CODE.keys()]), default=None)
    
    parser.add_argument("--temperature", default=0.0, help="Temperature to use for sampling", type=float)
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
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
        logging.getLogger("Whisper").setLevel(logging.DEBUG)

    output_dir = args.pop("output_dir")
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for audio_path in audio_files:

        result = transcribe_timestamped(
            model, audio_path,
            temperature=temperature,
            plot_word_alignment=plot_word_alignment,
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
                with open(outname + ".words.csv", "w", encoding="utf-8") as csv:
                    write_csv_words(result["segments"], file=csv)

        elif not args["verbose"]:

            json.dump(result, sys.stdout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    cli()