import logging
import numpy as np
# keep torch import if perform_word_alignment might need it internally
import torch
import re
import traceback
import math
import unicodedata
# removed: import gc

# use consistent imports relative to the package structure
from .audio.load import load_audio_and_resample, calculate_mel_for_segment
from .model.download import get_onnx_model_paths
from .model.load import load_onnx_models, get_tokenizer_for_model
from .align import (
    perform_word_alignment,
    _get_alignment_heads,
    _ALIGNMENT_HEADS,
    AUDIO_TIME_PER_TOKEN,
    SAMPLE_RATE, # use sample rate defined in align.py or here
)

logger = logging.getLogger("urro_whisper") # consistent logger name
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    # basic formatter, can be customized
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING) # default level

# constants for chunking
CHUNK_LENGTH_SAMPLES = 30 * SAMPLE_RATE  # 30 seconds * 16000 hz
CONTEXT_WORDS = 5  # number of words for textual context
MAX_CONTEXT_SEARCH_WORDS = 15 # how far back to look for valid timestamps for audio context
MIN_CHUNK_SAMPLES = int(0.2 * SAMPLE_RATE) # minimum audio length (e.g., 0.2s) to process a chunk

def whisperer(
    model: str,
    audio: str, # path or numpy array
    language: str = "en",
    # task parameter removed, always transcribe
    diarization_prefix: str = " -", # initial prefix only, critical for speaker segmentation
    disable_timestamps: bool = False,
    onnx_encoder: str = None,
    onnx_decoder: str = None,
    # enable_alignment parameter removed, always enabled
    verbose: bool = False,
    max_tokens: int = 448, # max total sequence length limit for decoder, including prompt/prefix
    onnx_providers: list = None,
    exclude_providers_on_error: list = ['CoreMLExecutionProvider'] # default excludes coreml
):
    """
    transcribes audio using whisper onnx models with chunking, context handling,
    and forced alignment.
    """
    # hardcode task to transcribe
    task = "transcribe"

    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    logger.info(f"starting transcription for: {audio if isinstance(audio, str) else 'numpy array'}".lower())
    # log message updated to remove task
    logger.info(f"using model shorthand: {model}, language: {language}".lower())

    # load and prepare audio
    if isinstance(audio, str):
        audio_data, actual_sr = load_audio_and_resample(audio, verbose=verbose)
    elif isinstance(audio, np.ndarray):
        audio_data = audio.astype(np.float32)
        actual_sr = SAMPLE_RATE # assume 16khz
        logger.info("using provided numpy array as audio input".lower())
        if audio.ndim > 1:
             logger.warning("input numpy array has >1 dimension, averaging channels".lower())
             audio_data = audio_data.mean(axis=-1) # ensure it becomes 1d
    else:
        raise TypeError("input 'audio' must be a file path (str) or a numpy array")

    if actual_sr != SAMPLE_RATE:
        logger.error(f"audio data sample rate ({actual_sr}) must be {SAMPLE_RATE}".lower())
        raise ValueError(f"incorrect audio sample rate: {actual_sr}")

    total_samples = len(audio_data)
    duration = total_samples / SAMPLE_RATE
    logger.info(f"total audio duration: {duration:.2f}s".lower())

    # load models and tokenizer
    encoder_path, decoder_path = get_onnx_model_paths(
        model, onnx_encoder, onnx_decoder, verbose=verbose
    )
    encoder_sess, decoder_sess = load_onnx_models(
        encoder_path, decoder_path, onnx_providers=onnx_providers, verbose=verbose,
        exclude_providers=exclude_providers_on_error # pass exclusion list
    )
    tokenizer, is_multilingual = get_tokenizer_for_model(model, language, task)
    eos_token = tokenizer.eot # <|endoftext|> token id
    n_mels_required = 128 if "large-v3" in model else 80 # determine n_mels based on model

    # initialize accumulators
    words = []
    transcript = ""
    tokens = []
    offset_samples = 0
    chunk_index = 0

    # initial state for chunk 1
    prefix = diarization_prefix # critical: initialize with the desired diarization prefix
    context = np.array([], dtype=np.float32) # simplified: context_audio -> context

    # main chunking loop
    while offset_samples < total_samples:
        # per-iteration variables (for clarity and cleanup)
        mel = None; enc_out = None; hidden_states = None; chunk_tokens = None
        decoding_error = None; output_tokens = None; text_token_ids = None
        chunk_words = []; align_trace = None; attentions = []; layer_attentions = []
        align_outs = None; new_tokens = []

        chunk_index += 1
        start_sample = offset_samples
        end_sample = min(offset_samples + CHUNK_LENGTH_SAMPLES, total_samples)
        chunk_duration = (end_sample - start_sample) / SAMPLE_RATE

        if (end_sample - start_sample) < MIN_CHUNK_SAMPLES and chunk_index > 1:
             logger.info(f"skipping very short final segment ({chunk_duration:.2f}s)".lower())
             break

        logger.info(f"processing chunk {chunk_index}: samples {start_sample}-{end_sample} ({chunk_duration:.2f}s)".lower())

        # prepare audio segment: context + current chunk
        chunk_audio = audio_data[start_sample:end_sample]
        segment = np.concatenate([context, chunk_audio])
        segment_dur = len(segment) / SAMPLE_RATE
        context_dur = len(context) / SAMPLE_RATE

        if len(segment) < MIN_CHUNK_SAMPLES: # check combined length
            logger.warning(f"chunk {chunk_index}: segment audio too short ({segment_dur:.2f}s) after adding context, skipping".lower())
            offset_samples = end_sample
            context = np.array([], dtype=np.float32)
            prefix = diarization_prefix # reset prefix
            # explicitly clean up before continuing
            del mel, enc_out, hidden_states, chunk_tokens, decoding_error
            del output_tokens, text_token_ids, chunk_words, align_trace
            del attentions, layer_attentions, align_outs, new_tokens
            # removed: gc.collect()
            continue

        # feature extraction
        try:
            mel = calculate_mel_for_segment(segment, model, n_mels_required, verbose=verbose)
        except Exception as e:
            logger.error(f"mel calculation failed for chunk {chunk_index}: {e}\n{traceback.format_exc()}".lower())
            offset_samples = end_sample; context = np.array([], dtype=np.float32); prefix = diarization_prefix
            del mel # removed: gc.collect()
            continue

        # run encoder
        enc_input_name = encoder_sess.get_inputs()[0].name
        try:
            enc_out = encoder_sess.run(None, {enc_input_name: mel.astype(np.float32)})
            hidden_states = enc_out[0]
            n_frames = hidden_states.shape[1]
        except Exception as e:
             logger.error(f"onnx encoder failed for chunk {chunk_index}: {e}\n{traceback.format_exc()}".lower())
             offset_samples = end_sample; context = np.array([], dtype=np.float32); prefix = diarization_prefix
             # explicitly clean up before continuing
             del mel, enc_out, hidden_states
             # removed: gc.collect()
             continue

        # prepare decoder prompt
        prompt_tokens = [tokenizer.sot]
        if is_multilingual:
            lang_token_str = f"<|{language}|>"
            try: prompt_tokens.append(tokenizer.encode(lang_token_str, allowed_special="all")[0])
            except Exception: pass # ignore if language token isn't found/needed

        # use hardcoded task "transcribe" for token
        task_token_str = "<|transcribe|>"
        try: prompt_tokens.append(tokenizer.encode(task_token_str, allowed_special="all")[0])
        except Exception as e: logger.error(f"failed to encode task token: {e}".lower()); raise
        if disable_timestamps:
            if tokenizer.no_timestamps is not None: prompt_tokens.append(tokenizer.no_timestamps)
        else: prompt_tokens.append(tokenizer.timestamp_begin) # use <|0.00|>

        # encode the current prefix text (critical for diarization forcing)
        try:
            prefix_tokens = tokenizer.encode(prefix, allowed_special="all")
            if not prefix_tokens: prefix_tokens = [] # handle empty prefix gracefully
            elif len(prefix_tokens) > 1 and verbose: logger.info(f"chunk {chunk_index}: multi-token prefix: {prefix_tokens} ('{prefix}')".lower())
        except Exception: prefix_tokens = [] # fallback to empty if encoding fails

        # combine initial prompt and the forced prefix tokens
        chunk_tokens = prompt_tokens[:] + prefix_tokens
        prompt_len = len(chunk_tokens) # length of prompt + forced prefix

        # run decoder generation loop
        dec_input_names = [inp.name for inp in decoder_sess.get_inputs()]
        dec_output_names = [out.name for out in decoder_sess.get_outputs()]
        ids_name = "input_ids"
        states_name = "encoder_hidden_states"
        logits_name = "logits"

        # calculate max new tokens allowed based on current prompt/prefix length
        max_new_tokens = max_tokens - len(chunk_tokens)

        if verbose: logger.info(f"chunk {chunk_index}: starting decoder generation (max new tokens={max_new_tokens}, total len limit={max_tokens})".lower())
        decoding_error = None
        # capture generated tokens (excluding eos) in a separate list for this chunk
        new_tokens = []

        try:
            if max_new_tokens <= 0:
                 logger.warning(f"chunk {chunk_index}: prompt/prefix length ({len(chunk_tokens)}) meets or exceeds max_tokens ({max_tokens}), no tokens generated".lower())
            else:
                current_gen_tokens = chunk_tokens[:] # copy for generation loop
                for step in range(max_new_tokens):
                    input_ids_np = np.array([current_gen_tokens], dtype=np.int64)
                    decoder_inputs = {
                        ids_name: input_ids_np,
                        states_name: hidden_states.astype(np.float32)
                    }
                    # only need logits for greedy decoding
                    decoder_outputs = decoder_sess.run([logits_name], decoder_inputs)

                    logits = decoder_outputs[0]
                    next_token_logits = logits[0, -1, :]
                    next_token = int(np.argmax(next_token_logits))

                    if next_token == eos_token:
                        if verbose: logger.info(f"chunk {chunk_index}: eos token ({eos_token}) detected at step {step+1}".lower())
                        break # stop decoding for this chunk

                    current_gen_tokens.append(next_token)
                    new_tokens.append(next_token) # store only the newly generated tokens

                # check if loop finished by reaching max steps
                if step == max_new_tokens - 1 and next_token != eos_token:
                     logger.info(f"chunk {chunk_index}: reached max generated token limit ({max_new_tokens})".lower())
                # update chunk_tokens to include the generated ones for alignment context
                chunk_tokens = current_gen_tokens[:]

        except Exception as e:
            decoding_error = e
            logger.error(f"error during decoder generation loop for chunk {chunk_index}: {decoding_error}\n{traceback.format_exc()}".lower())

        # append the valid generated tokens from this chunk to the global list
        tokens.extend(new_tokens)

        # post-processing and alignment
        chunk_words = [] # re-initialize for safety
        align_trace = None

        try:
            # get tokens generated after the initial prompt and forced prefix
            output_tokens = new_tokens # use the tokens actually generated in this chunk
            # filter out special/timestamp tokens for text content
            text_token_ids = [t for t in output_tokens if t < tokenizer.sot]

            if not text_token_ids and decoding_error is None:
                 logger.warning(f"chunk {chunk_index}: no text tokens generated after prompt/prefix".lower())

            elif text_token_ids:
                # --- alignment ---
                hidden_size = hidden_states.shape[-1]
                size_map = {384: "tiny", 512: "base", 768: "small", 1024: "medium", 1280: "large"}
                model_base = size_map.get(hidden_size, model.split("-")[0].split(".")[0]) # try to infer base model name
                # construct full name for alignment head lookup
                model_name = (f"{model_base}.en" if (language == "en" and f"{model_base}.en" in _ALIGNMENT_HEADS) else model_base)
                if model in _ALIGNMENT_HEADS: model_name = model # use exact model name if available

                # check if decoder outputs cross-attentions
                attn_names = [name for name in dec_output_names if "cross_attentions" in name]
                # alignment check updated: removed enable_alignment condition
                can_align = bool(attn_names)
                attentions = [] # re-initialize inner scope variable

                if can_align:
                    # determine decoder layers and heads for alignment
                    num_layers = 0; max_layer_idx = -1
                    for name in attn_names: match = re.search(r"\.(\d+)", name);
                    if match: max_layer_idx = max(max_layer_idx, int(match.group(1)))
                    num_layers = max_layer_idx + 1
                    if num_layers == 0: # fallback if regex fails
                         layer_map = {"tiny": 4, "base": 6, "small": 12, "medium": 24, "large": 32}; num_layers = layer_map.get(model_base, 6)
                    num_heads = hidden_size // 64 # standard whisper head size
                    align_heads = _get_alignment_heads(model_name, num_layers, num_heads)

                    if align_heads is None: logger.warning(f"could not load alignment heads for {model_name}, alignment may be less accurate".lower())
                    if verbose: logger.info(f"chunk {chunk_index}: extracting cross-attentions...".lower())

                    layer_attentions = [[] for _ in range(num_layers)] # re-initialize inner scope variable
                    align_failed = False
                    # use prompt+prefix as the initial input for attention extraction
                    align_tokens_input = chunk_tokens[:prompt_len]
                    # tokens to extract attention for
                    valid_text_tokens = text_token_ids

                    try: # wrap attention extraction in try/except
                        for token_idx, token_id in enumerate(valid_text_tokens):
                            # fix: input for getting attention for `token_id` should not include `token_id` itself yet.
                            # the input represents the state before generating token_id.
                            align_ids_np = np.array([align_tokens_input], dtype=np.int64)
                            align_inputs = {
                                ids_name: align_ids_np,
                                states_name: hidden_states.astype(np.float32)
                            }
                            # request all cross-attention outputs
                            align_req_outputs = attn_names
                            try: # try/except around the run call itself
                                align_outs = decoder_sess.run(align_req_outputs, align_inputs)
                                for layer_idx, att_tensor in enumerate(align_outs):
                                    # attention is for the last token prediction based on input
                                    layer_attentions[layer_idx].append(att_tensor[0, :, -1, :]) # [heads, key_len (n_frames)]
                            except Exception as e_inner:
                                align_failed = True; logger.error(f"alignment pass run failed for token {token_idx}: {e_inner}".lower()); break
                            # now append the current token for the next iteration's input state
                            align_tokens_input.append(token_id)


                        if not align_failed and all(len(lst) == len(valid_text_tokens) for lst in layer_attentions):
                            try: # stack attentions for each layer -> [1, heads, seq_len, key_len]
                                attentions = [np.stack(layer_atts, axis=1)[np.newaxis, :, :, :] for layer_atts in layer_attentions]
                            except Exception as e_stack: logger.error(f"stacking attentions failed: {e_stack}".lower()); attentions = []
                        else:
                             if not align_failed: logger.warning(f"chunk {chunk_index}: attention length mismatch (expected {len(valid_text_tokens)}, got varying lengths)".lower())
                             attentions = []
                    except Exception as e_outer:
                         logger.error(f"error during attention extraction setup: {e_outer}".lower())
                         attentions = [] # ensure it's empty

                # perform dtw alignment
                if attentions:
                    if verbose: logger.info(f"chunk {chunk_index}: performing dtw alignment".lower())
                    try: # wrap alignment call
                        # pass the full sequence including prompt/prefix and generated tokens
                        # but align only the generated text tokens
                        raw_chunk_words = perform_word_alignment(
                            full_token_sequence=chunk_tokens, # the full sequence including prompt+generated
                            generated_text_tokens_ids=valid_text_tokens, # only the text tokens generated after prompt
                            cross_attentions_list=attentions,
                            tokenizer=tokenizer, alignment_heads=align_heads,
                            model_n_text_layers=num_layers,
                            n_frames_feature=n_frames, language=language,
                            medfilt_width=7, qk_scale=1.0, debug=verbose
                        )
                        if raw_chunk_words:
                            if verbose: logger.info(f"chunk {chunk_index}: dtw successful".lower())
                            chunk_start_time = start_sample / SAMPLE_RATE
                            for wt in raw_chunk_words:
                                start_rel, end_rel = wt.get('start'), wt.get('end')
                                start_glob, end_glob = None, None
                                if start_rel is not None:
                                    # adjust timestamps relative to the start of the current chunk's audio
                                    if start_rel >= context_dur: # only include words starting after context audio
                                        start_glob = round(max(0, start_rel - context_dur) + chunk_start_time, 3)
                                    else: continue # skip words starting during context audio
                                if end_rel is not None:
                                    # adjust end time relative to the start of the current chunk's audio
                                    end_glob = round(max(0, end_rel - context_dur) + chunk_start_time, 3)
                                    # sanity check: ensure end is not before start
                                    if start_glob is not None and end_glob < start_glob: end_glob = start_glob
                                # changed 'word' key to 'text'
                                if wt.get('word'): # only add if word exists
                                    chunk_words.append({"text": wt['word'], "start": start_glob, "end": end_glob})
                        else: logger.warning(f"chunk {chunk_index}: dtw alignment failed or returned empty".lower())
                    except Exception as e_dtw:
                        logger.error(f"error during perform_word_alignment call: {e_dtw}\n{traceback.format_exc()}".lower())
                        chunk_words = [] # ensure empty on alignment error
                else:
                     # message updated - alignment always attempted if possible
                     logger.warning(f"chunk {chunk_index}: alignment not performed (no cross-attentions or extraction failed)".lower())

                # fallback for context prep if alignment failed but we have text
                if not chunk_words and valid_text_tokens:
                     chunk_text_only = tokenizer.decode(valid_text_tokens).strip()
                     words_only = chunk_text_only.split()
                     # create dummy timestamp dicts for context prep
                     # changed 'word' key to 'text'
                     chunk_words = [{"text": w, "start": None, "end": None} for w in words_only]
                     logger.warning(f"chunk {chunk_index}: using basic text split for context prep due to alignment failure".lower())

        except Exception as e_post:
            logger.error(f"error during post-processing/alignment setup for chunk {chunk_index}: {e_post}".lower())
            align_trace = traceback.format_exc()
            chunk_words = [] # ensure empty on error

        # aggregate results
        # only add words with valid start timestamps to the global list
        valid_words = [wt for wt in chunk_words if wt.get('start') is not None]
        words.extend(valid_words)
        # get text parts from this chunk's words (aligned or fallback)
        # changed 'word' key to 'text'
        chunk_text = [wt['text'] for wt in chunk_words if wt.get('text')]
        if chunk_text: transcript += " ".join(chunk_text) + " "

        # prepare context for next chunk (critical for diarization consistency)
        context = np.array([], dtype=np.float32) # reset context audio
        next_prefix = diarization_prefix

        if chunk_words:
            # 1. prepare text context (last few words)
            # changed 'word' key to 'text'
            context_word_data = [wt for wt in chunk_words if wt.get('text')][-CONTEXT_WORDS:]
            context_text = [wt['text'] for wt in context_word_data]

            # 2. find last speaker prefix (e.g., "-", ">>") for diarization continuity
            speaker_prefix = diarization_prefix
            potential_prefixes = ["-", ">>", "<"] # add other prefixes if needed
            # search back reasonably far in this chunk's words
            prefix_search_limit = min(len(chunk_words), MAX_CONTEXT_SEARCH_WORDS * 2)
            for wt in reversed(chunk_words[-prefix_search_limit:]):
                 # changed 'word' key to 'text'
                word_stripped = wt.get('text', '').strip()
                if word_stripped in potential_prefixes:
                    speaker_prefix = word_stripped + " " # include space after prefix
                    break # found the most recent prefix

            # clean up context text parts (remove punctuation from last word if needed)
            if context_text:
                 last_word = context_text[-1]
                 # remove common punctuation, keep core word
                 clean_last_word = "".join(c for c in last_word if not unicodedata.category(c).startswith('P')).strip()
                 if clean_last_word: context_text[-1] = clean_last_word
                 elif len(context_text) > 1: context_text.pop() # remove if last word was only punctuation

            # combine prefix and text for next chunk's prompt
            next_prefix = (speaker_prefix + " ".join(context_text)).strip()
            if not next_prefix: next_prefix = diarization_prefix # ensure it's never empty

            # 3. prepare audio context (based on timestamps of context words)
            first_ctx_word = None; last_ctx_word = None
            # search back for words with valid timestamps
            ts_search_limit = min(len(chunk_words), MAX_CONTEXT_SEARCH_WORDS)
            ts_ctx_words = [wt for wt in chunk_words[-ts_search_limit:] if wt.get('start') is not None and wt.get('end') is not None]

            if ts_ctx_words:
                 last_ctx_word = ts_ctx_words[-1]
                 # find the start word corresponding to the textual context
                 start_index_in_filtered = max(0, len(ts_ctx_words) - CONTEXT_WORDS)
                 first_ctx_word = ts_ctx_words[start_index_in_filtered]

            if first_ctx_word and last_ctx_word:
                # get audio slice based on the global timestamps of the context words
                ctx_start_sample = max(0, math.floor(first_ctx_word['start'] * SAMPLE_RATE))
                ctx_end_sample = min(total_samples, math.ceil(last_ctx_word['end'] * SAMPLE_RATE))

                if ctx_end_sample > ctx_start_sample:
                    context = audio_data[ctx_start_sample:ctx_end_sample]
                    # limit context audio length (e.g., 5 seconds max)
                    max_context_s = 5.0
                    if len(context) > max_context_s * SAMPLE_RATE:
                        logger.warning(f"trimming context audio > {max_context_s}s".lower())
                        context = context[-int(max_context_s * SAMPLE_RATE):]
                    prefix = next_prefix # set prefix for next chunk
                    if verbose: logger.info(f"chunk {chunk_index}: prepared context audio ({len(context)/SAMPLE_RATE:.2f}s) and prefix '{prefix}'".lower())
                else:
                    logger.warning(f"empty context audio slice calculated, resetting context".lower())
                    context = np.array([], dtype=np.float32); prefix = diarization_prefix
            else:
                logger.warning(f"chunk {chunk_index}: no valid audio context timestamps found, resetting context".lower())
                context = np.array([], dtype=np.float32); prefix = diarization_prefix
        else:
             # no words generated or alignment failed completely
             if decoding_error is None: logger.warning(f"chunk {chunk_index}: no words available, resetting context".lower())
             context = np.array([], dtype=np.float32); prefix = diarization_prefix

        # move to the next chunk start position
        offset_samples = end_sample

        # explicit cleanup at end of loop iteration
        del mel, enc_out, hidden_states, chunk_tokens, decoding_error
        del output_tokens, text_token_ids, chunk_words, align_trace
        del new_tokens # delete chunk-specific token list
        # delete potentially large alignment variables if they exist
        if 'attentions' in locals(): del attentions
        if 'layer_attentions' in locals(): del layer_attentions
        if 'align_outs' in locals(): del align_outs
        # removed: gc.collect()
        if verbose: logger.info(f"--- end of chunk {chunk_index} processing".lower())

    # final result combination
    final_result = {
        "text": transcript.strip(), # simplified: full_transcript_text -> transcript
        "words": words,           # simplified: all_word_timestamps -> words
        "tokens": tokens,          # simplified: all_output_token_ids -> tokens
        "language": language,
        # alignment_method removed as it's always enabled now
        # audio_data removed to avoid returning large array
    }

    return final_result
