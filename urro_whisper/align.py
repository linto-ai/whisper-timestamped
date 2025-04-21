import numpy as np
import torch
import logging
import gzip
import base64
import traceback

# alignment specific imports
try:
    import dtw
    from scipy.ndimage import median_filter
except ImportError:
    raise ImportError("please install alignment dependencies: `pip install dtw-python scipy`")

# icu tokenizer import for better word splitting, with fallback
try:
    import icu
    from icu import Locale, BreakIterator
    import unicodedata
    icu_available = True
except ImportError:
    icu_available = False
    logging.warning("pyicu not found, word splitting will rely on basic space/punctuation splitting. install with `pip install pyicu`")

logger = logging.getLogger("urro_whisper") # consistent logger name
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # set a default level if not configured elsewhere
    logger.setLevel(logging.WARNING)


# constants
# define audio constants used in timestamp calculations
SAMPLE_RATE = 16000
HOP_LENGTH = 160 # from whisper feature extraction
AUDIO_SAMPLES_PER_TOKEN = HOP_LENGTH * 2 # each token corresponds to 2 feature frames
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / SAMPLE_RATE # time duration of one token

_punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" # common punctuation from whisper tokenizer

# alignment head data (precomputed masks)
_ALIGNMENT_HEADS = {
    # standard models - compressed boolean masks indicating which attention heads are good for alignment
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b'ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj',
    "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    # aliases
    "large": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00", # large defaults to v3
    # turbo models (distil-large-v3)
    "large-v3-turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
    "turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
}

def _get_alignment_heads(model_name, num_layers, num_heads):
    """retrieves and validates the alignment head mask for a given model configuration"""
    if model_name not in _ALIGNMENT_HEADS:
        logger.warning(f"alignment heads not found for model '{model_name}', using default behavior (all heads averaged). timestamps might be less accurate")
        return None # fallback: average all heads later

    dump = _ALIGNMENT_HEADS[model_name]
    try:
        # decompress and decode the mask
        array = np.frombuffer(gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()
        expected_size = num_layers * num_heads
        if array.size != expected_size:
            # this indicates a mismatch between the compiled mask and the model's reported layers/heads
            logger.warning(f"alignment head data size mismatch for {model_name}. expected {expected_size} ({num_layers}x{num_heads}), got {array.size}. using None")
            return None
        mask = torch.from_numpy(array).reshape(num_layers, num_heads)
        alignment_heads = mask.to_sparse() # convert to sparse tensor for easier indexing
        logger.info(f"loaded alignment heads for {model_name} with shape: ({num_layers}, {num_heads})")
        return alignment_heads
    except Exception as e:
        logger.error(f"error processing alignment heads for {model_name}: {e}. using None")
        return None


# icu/fallback word splitting

def is_letter_token_icu(token_str):
    """check if a token string contains at least one unicode letter character"""
    if not icu_available:
        return any(c.isalpha() for c in token_str) # basic fallback if icu is not installed
    for char in token_str:
        # check unicode category, 'l' denotes letters
        if unicodedata.category(char).startswith('L'):
            return True
    return False

def split_tokens_on_unicode_fallback(text_tokens: list, tokenizer):
    """basic fallback word splitting based on spaces and punctuation if pyicu is unavailable"""
    words = []
    word_tokens_list = [] # stores token ids for each word
    word_indices_list = [] # stores original indices (relative to text_tokens) for each word's tokens
    current_tokens = []
    current_indices = []
    current_word = ""

    for i, token_id in enumerate(text_tokens):
        token_str = tokenizer.decode([token_id])
        token_str_strip = token_str.strip()
        starts_with_space = token_str.startswith(' ')
        # check if the stripped token is purely punctuation
        is_punctuation_only = token_str_strip and all(c in _punctuation for c in token_str_strip)
        is_first_token = not current_word

        # decide whether to split before this token
        should_split = (starts_with_space and not is_first_token) or \
                       (is_punctuation_only and not is_first_token) # split on space or if token is just punctuation (unless it's the start)

        if should_split and current_word:
            # store the completed word and its tokens/indices
            words.append(current_word)
            word_tokens_list.append(current_tokens[:])
            word_indices_list.append(current_indices[:])
            # reset for the next word
            current_tokens = []
            current_indices = []
            current_word = ""

        # append the current token to the potential next word
        if token_str_strip: # avoid adding empty strings from spaces
            if not (starts_with_space and should_split and not is_first_token): # don't add the space itself if it caused a split
                current_tokens.append(token_id)
                current_indices.append(i)
                current_word += token_str # add the full token string (including leading space if any)
                current_word = current_word.strip() # keep the internal word representation clean

    # add the last accumulated word
    if current_word:
        words.append(current_word)
        word_tokens_list.append(current_tokens)
        word_indices_list.append(current_indices)

    # filter out words that become empty after stripping punctuation (e.g., a word that was just "-")
    filtered_words = []
    filtered_tokens = []
    filtered_indices = []
    for i, word in enumerate(words):
        word_strip = word.strip(_punctuation + ' ') # strip leading/trailing spaces and punctuation
        if word_strip: # only keep if something remains
            filtered_words.append(word) # keep original word form
            filtered_tokens.append(word_tokens_list[i])
            filtered_indices.append(word_indices_list[i])

    return filtered_words, filtered_tokens, filtered_indices


def split_tokens_with_icu(text_tokens: list, tokenizer, lang='en'):
    """
    splits tokens into words using icu word boundaries and maps tokens to words.
    filters out tokens/words that don't contain letters for alignment purposes.
    """
    if not icu_available:
        logger.warning("using fallback word splitting due to missing pyicu")
        return split_tokens_on_unicode_fallback(text_tokens, tokenizer)

    # decode the relevant text tokens for icu processing
    decoded_text = tokenizer.decode(text_tokens).strip()
    if not decoded_text: return [], [], []

    try:
        locale = Locale(lang) # create icu locale for the specified language
        breaker = BreakIterator.createWordInstance(locale) # create word boundary iterator
    except Exception as e:
        logger.error(f"failed to create icu breakiterator for lang='{lang}', error: {e}. using fallback splitting")
        return split_tokens_on_unicode_fallback(text_tokens, tokenizer)

    breaker.setText(decoded_text)

    words = [] # list of identified words
    word_tokens_list = [] # list of token id lists, one per word
    word_indices_list = [] # list of original token index lists, one per word

    token_spans = [] # stores (start_offset, end_offset) in decoded_text for each *letter-containing* token
    running_offset = 0 # track position in decoded_text
    filtered_indices_map = {} # map original token index -> filtered token index (for letter tokens only)
    filtered_text_tokens = [] # list of token ids containing letters
    filtered_original_indices = [] # original indices corresponding to filtered_text_tokens
    current_filtered_idx = 0

    # first pass: identify letter tokens and their approximate spans in the decoded text
    for original_idx, token_id in enumerate(text_tokens):
        token_str = tokenizer.decode([token_id])
        token_str_strip = token_str.lstrip() # use lstrip for finding offset, but keep original token_str

        # only consider tokens with letters for alignment mapping
        if is_letter_token_icu(token_str_strip):
            filtered_text_tokens.append(token_id)
            filtered_original_indices.append(original_idx)
            filtered_indices_map[original_idx] = current_filtered_idx
            current_filtered_idx += 1

            # find the token's position in the decoded string
            try:
                # search for the exact token string first
                offset = decoded_text.index(token_str, running_offset)
                # crude check for large jump, might indicate error due to normalization/whitespace differences
                if offset > running_offset + 10:
                     # try finding the stripped version if exact match jumped too far
                     offset_strip = decoded_text.find(token_str_strip, running_offset)
                     if offset_strip != -1 and abs(offset_strip - running_offset) < abs(offset - running_offset):
                          offset = offset_strip
                          token_len = len(token_str_strip)
                     else:
                          token_len = len(token_str) # stick with original if stripped wasn't better
                else:
                     token_len = len(token_str)
            except ValueError: # if exact token not found
                try: # try finding the stripped version
                    offset = decoded_text.index(token_str_strip, running_offset)
                    token_len = len(token_str_strip)
                except ValueError: # if still not found, log warning and approximate
                    logger.warning(f"could not accurately map token '{token_str}' (id: {token_id}) to text offset, using approximate position")
                    offset = running_offset # place it at the current tracked offset
                    token_len = len(token_str_strip) # use stripped length as best guess

            token_spans.append((offset, offset + token_len))
            running_offset = offset + token_len # advance offset
        else:
            # even for non-letter tokens, advance the running offset roughly
            try:
                offset = decoded_text.index(token_str, running_offset)
                running_offset = offset + len(token_str)
            except ValueError:
                 try: # try stripped version if original fails
                      offset = decoded_text.index(token_str_strip, running_offset)
                      running_offset = offset + len(token_str_strip)
                 except ValueError: pass # ignore if token can't be found at all
            continue # skip adding non-letter tokens to spans

    # second pass: iterate through icu word boundaries and map tokens to words
    start = breaker.first()
    for end in breaker:
        word = decoded_text[start:end]
        word_stripped = word.strip()

        # skip whitespace or words without letters (e.g., punctuation-only words)
        if not word_stripped or not is_letter_token_icu(word_stripped):
            start = end
            continue

        current_word_token_original_indices = []
        word_start = start # character offset of the word start
        word_end = end # character offset of the word end

        # find which *letter-containing* tokens overlap with this word's span
        for i, (tok_start, tok_end) in enumerate(token_spans):
            # simple overlap check: token starts before word ends and token ends after word starts
            if tok_start < word_end and tok_end > word_start:
                # get the original index of this token (relative to the input text_tokens list)
                original_token_idx = filtered_original_indices[i]
                current_word_token_original_indices.append(original_token_idx)

        # if we found tokens belonging to this word
        if current_word_token_original_indices:
            words.append(word_stripped) # store the cleaned word
            # get the original token indices that are also *letter-containing* tokens
            relative_indices = [idx for idx in current_word_token_original_indices if idx in filtered_indices_map]
            # store the original indices (relative to input text_tokens) and the corresponding token ids
            word_indices_list.append(relative_indices)
            word_tokens_list.append([text_tokens[i] for i in relative_indices])

        start = end # move to the next word boundary

    return words, word_tokens_list, word_indices_list


def perform_word_alignment(
    full_token_sequence, # includes prompt, prefix, generated tokens
    generated_text_tokens_ids, # only the generated text tokens (no specials)
    cross_attentions_list, # list of numpy arrays [batch, heads, seq_len, key_len] per layer
    tokenizer,
    alignment_heads, # sparse tensor mask from _get_alignment_heads
    model_n_text_layers, # number of decoder layers
    n_frames_feature, # number of audio frames in the encoder output (mel spectrogram time dim)
    medfilt_width=7, # median filter width for attention smoothing
    qk_scale=1.0, # scaling factor for attention before softmax (usually 1.0)
    debug=False, # more verbose logging during alignment
    language='en' # language for word splitting
):
    """
    performs dtw alignment between text tokens and audio frames using cross-attentions.
    returns a list of dictionaries, each containing 'word', 'start', 'end'.
    """
    if not cross_attentions_list:
         logger.warning("no cross-attentions provided for alignment, cannot perform dtw alignment")
         return []
    if not generated_text_tokens_ids:
         logger.warning("no generated text tokens provided for alignment")
         return []

    # attention processing
    if len(cross_attentions_list) != model_n_text_layers:
         logger.warning(f"expected {model_n_text_layers} cross-attention layers, but got {len(cross_attentions_list)}, using available layers")
         model_n_text_layers = len(cross_attentions_list) # adjust layer count to match data

    # stack attentions: list of (batch, heads, seq_len, key_len) -> (layers, batch, heads, seq_len, key_len)
    try:
        # convert numpy arrays from onnx output to torch tensors
        relevant_attentions = torch.stack([torch.from_numpy(att) for att in cross_attentions_list])
    except Exception as e:
        logger.error(f"error stacking attention tensors: {e}")
        return []

    # handle batch dimension (should ideally be 1)
    if relevant_attentions.shape[1] == 1:
        weights = relevant_attentions.squeeze(1) # shape: (layers, heads, seq_len, key_len/n_frames)
    elif relevant_attentions.shape[1] > 1:
        logger.warning(f"attention batch size is {relevant_attentions.shape[1]}, expected 1, using first batch element")
        weights = relevant_attentions[:, 0, :, :, :]
    else:
        logger.error(f"invalid attention batch size: {relevant_attentions.shape[1]}")
        return []

    # sequence length from attention tensors (should match generated_text_tokens_ids)
    seq_len_att = weights.shape[2]
    num_text_tokens = len(generated_text_tokens_ids)

    # sanity check and potential trimming if lengths mismatch (can happen with beam search?)
    if seq_len_att != num_text_tokens:
         logger.warning(f"attention sequence length ({seq_len_att}) differs from generated text token count ({num_text_tokens}) for alignment, using {min(seq_len_att, num_text_tokens)} tokens")
         min_len = min(seq_len_att, num_text_tokens)
         # trim attention sequence length to match the shorter one
         weights = weights[:, :, :min_len, :]
         # don't trim generated_text_tokens_ids here; split_tokens will handle the full list later
         # update the effective sequence length used
         seq_len_att = min_len
         if seq_len_att == 0:
              logger.error("no text tokens remain after adjusting for attention length mismatch")
              return []

    # apply alignment heads mask if available and valid
    if alignment_heads is not None and alignment_heads.shape[0] == model_n_text_layers and alignment_heads.shape[1] == weights.shape[1]:
        selected_weights = []
        # get [layer, head] indices from the sparse mask
        head_indices = alignment_heads.indices().T.tolist()
        for layer_idx, head_idx in head_indices:
            # ensure indices are within the bounds of the actual weights tensor
            if layer_idx < weights.shape[0] and head_idx < weights.shape[1]:
                 # select weights for this specific head: [seq_len_att, n_frames_feature]
                 selected_weights.append(weights[layer_idx, head_idx, :, :n_frames_feature])
            else:
                 logger.warning(f"alignment head index ({layer_idx}, {head_idx}) out of bounds for weights shape {weights.shape[:2]}")

        if not selected_weights:
             # if filtering resulted in no heads (e.g., all out of bounds), fallback to averaging all
             logger.warning("no valid alignment heads found/selected after filtering, averaging all heads/layers")
             weights = weights[:, :, :, :n_frames_feature].mean(dim=(0, 1)) # average across layers and heads -> [seq_len_att, n_frames_feature]
        else:
             # average the weights from the selected heads
             weights = torch.stack(selected_weights).mean(dim=0) # shape: [seq_len_att, n_frames_feature]
    else: # alignment heads not specified, invalid, or shape mismatch
        if alignment_heads is not None: # log if mismatch occurred
             logger.warning(f"alignment head shape mismatch ({alignment_heads.shape} vs layers={model_n_text_layers}, heads={weights.shape[1]}), averaging all heads/layers")
        else: # log if no heads were provided (this is expected if using default)
            logger.info("no alignment heads specified, averaging all heads/layers")
        # average across all layers and heads
        weights = weights[:, :, :, :n_frames_feature].mean(dim=(0, 1)) # shape: [seq_len_att, n_frames_feature]

    # validate weights shape before proceeding to dtw
    if weights.ndim != 2 or weights.shape[0] != seq_len_att or weights.shape[1] > n_frames_feature:
         logger.error(f"unexpected attention weights shape after processing: {weights.shape}, expected ({seq_len_att}, <= {n_frames_feature}), cannot proceed with dtw")
         return []

    # convert final weights to numpy for filtering and dtw
    text_token_weights = weights.float().cpu().numpy()

    if text_token_weights.shape[0] == 0:
         logger.warning("no attention weights found for text tokens after processing, cannot align")
         return []

    # apply median filter along the audio frame axis to smooth attention spikes
    if medfilt_width > 0 and text_token_weights.shape[1] > medfilt_width:
        if debug: logger.info(f"applying median filter with width {medfilt_width}")
        text_token_weights = median_filter(text_token_weights, (1, medfilt_width)) # filter each token's attention independently

    # apply softmax scaling (temperature scaling) along the audio frame axis
    if debug: logger.info(f"applying softmax scaling with qk_scale={qk_scale}")
    # multiplying by qk_scale increases/decreases the peakiness of the distribution
    text_token_weights = torch.from_numpy(text_token_weights * qk_scale).softmax(dim=-1).numpy()

    # prepare cost matrix for dtw (use negative weights: higher attention = lower cost)
    # cost_matrix shape: [num_text_tokens, num_audio_frames]
    cost_matrix = -text_token_weights

    # dtw alignment
    try:
        if debug: logger.info(f"running dtw on cost matrix shape: {cost_matrix.shape}")
        # dtw aligns text tokens (query, index1) to audio frames (template, index2)
        alignment = dtw.dtw(cost_matrix.astype(np.double), # requires float64
                            keep_internals=False, # don't need intermediate matrices
                            step_pattern=dtw.stepPattern.symmetric1) # standard step pattern
        if debug: logger.info("dtw finished")
    except Exception as e:
        logger.error(f"error during dtw: {e}, cannot generate word timestamps")
        if debug: traceback.print_exc()
        return [] # return empty list on dtw failure

    # extract the alignment path indices
    path_token_indices = alignment.index1 # indices into the text tokens used in dtw (0 to seq_len_att-1)
    path_frame_indices = alignment.index2 # indices into the audio frames (0 to n_frames_feature-1)

    # find frame boundaries where the aligned token index changes
    # np.diff finds changes, > 0 means the token index increased
    token_change_indices = np.where(np.diff(path_token_indices) > 0)[0] + 1
    # get the frame index corresponding to each token boundary
    # pad with frame 0 at the start to represent the beginning of the first token
    token_boundaries_frames = np.pad(path_frame_indices[token_change_indices], (1, 0), constant_values=0)

    # validate boundary count - should be #tokens + 1
    expected_boundaries = seq_len_att + 1 # number of tokens used in cost matrix + 1
    if len(token_boundaries_frames) != expected_boundaries:
         logger.warning(f"dtw boundary count ({len(token_boundaries_frames)}) mismatch with expected token count ({seq_len_att}), adjusting boundaries")
         # attempt to fix boundary array length if it's too short or too long
         if len(token_boundaries_frames) < expected_boundaries:
              diff = expected_boundaries - len(token_boundaries_frames)
              # pad with the last known frame index (or max frames if empty)
              last_val = token_boundaries_frames[-1] if len(token_boundaries_frames) > 0 else n_frames_feature
              token_boundaries_frames = np.pad(token_boundaries_frames, (0, diff), constant_values=last_val)
         else: # too long
              token_boundaries_frames = token_boundaries_frames[:expected_boundaries] # truncate
         # ensure the very last boundary doesn't exceed the number of frames
         token_boundaries_frames[-1] = min(token_boundaries_frames[-1], n_frames_feature)

    # map token boundaries to word boundaries
    # split the *original* generated text tokens (before potential length mismatch trim) into words
    # this uses the icu/fallback splitter to get word strings and their corresponding original token indices
    words, word_tokens_list, word_indices_list_rel = split_tokens_with_icu(
        generated_text_tokens_ids[:], tokenizer, lang=language # pass a copy
    )
    if debug: logger.info(f"split into {len(words)} words using icu/fallback")

    # calculate anchor time
    # whispers predicts timestamps relative to the start of the *audio segment fed to the encoder*
    # however, it sometimes includes timestamp tokens from the prompt/prefix in its output
    # we need to find the time offset represented by the *last* timestamp token *before* the actual generated text starts
    timestamp_boundaries = {} # map token index in full_token_sequence to its time value
    current_ts = 0.0
    prompt_len = 0 # heuristic length of the initial prompt (<|sot|><|lang|><|task|><|startoflm|>...)
    found_task_token = False

    # find end of prompt (heuristic: first special or timestamp token after sot)
    # also find all timestamp tokens and their values
    for i, token_id in enumerate(full_token_sequence):
        is_special = token_id >= tokenizer.sot
        is_timestamp = token_id >= tokenizer.timestamp_begin and token_id < tokenizer.eot # eot is not a timestamp

        # assume prompt ends *before* the first task or timestamp token is encountered
        if not found_task_token and (token_id == tokenizer.task or token_id == tokenizer.transcribe or token_id == tokenizer.translate or is_timestamp):
             prompt_len = i # index *before* this token
             found_task_token = True

        # store timestamp value if found
        if is_timestamp:
            current_ts = round((token_id - tokenizer.timestamp_begin) * AUDIO_TIME_PER_TOKEN, 3) # use 3 decimal places
            timestamp_boundaries[i] = current_ts

    # find the index of the first *actual text* token after the initial prompt and forced prefix
    first_gen_text_token_original_idx = -1
    # search starts after the estimated prompt length plus the known prefix length (which was forced)
    # note: prefix length calculation needs care if prefix itself contained special tokens
    search_start_index = prompt_len # start searching right after the determined prompt
    for i in range(search_start_index, len(full_token_sequence)):
        token_id = full_token_sequence[i]
        # check if it's a regular text token (not special, not timestamp)
        if token_id < tokenizer.sot and token_id < tokenizer.timestamp_begin:
            first_gen_text_token_original_idx = i
            break # found the first one

    # determine the anchor time based on the last timestamp before the first generated text token
    anchor_time = 0.0
    if first_gen_text_token_original_idx != -1:
         # find indices of timestamp tokens that occurred *before* the first text token
         relevant_ts_indices = sorted([idx for idx in timestamp_boundaries if idx < first_gen_text_token_original_idx])
         if relevant_ts_indices:
              last_ts_index = relevant_ts_indices[-1]
              anchor_time = timestamp_boundaries[last_ts_index]
              if debug: logger.info(f"found timestamp token {full_token_sequence[last_ts_index]} ({tokenizer.decode([full_token_sequence[last_ts_index]])}) at index {last_ts_index} ({anchor_time:.3f}s) before first text token index {first_gen_text_token_original_idx}")
         else: # no timestamp found before text started
              if debug: logger.info("no timestamp token found before the first generated text token, using anchor time 0.0s")
    else: # couldn't find the first generated text token (e.g., only special tokens generated)
         if debug: logger.info("could not determine the first generated text token index, using anchor time 0.0s")

    logger.info(f"timestamp anchor time set to: {anchor_time:.3f}s")

    # generate word timestamps
    word_timestamps_aligned = []
    # track which original token indices (relative to generated_text_tokens_ids) were successfully aligned to a word
    aligned_token_original_indices = set()

    for i, word in enumerate(words):
        # word_indices_list_rel contains indices relative to generated_text_tokens_ids
        relative_indices_for_word = word_indices_list_rel[i]
        if not relative_indices_for_word: continue # skip if word somehow has no tokens

        # filter these indices to only include those that were actually used in the dtw alignment
        # (i.e., indices from 0 up to seq_len_att - 1)
        dtw_indices_for_word = [idx for idx in relative_indices_for_word if idx < seq_len_att]

        if not dtw_indices_for_word:
            # this word consists entirely of tokens that were outside the dtw range (e.g., due to length mismatch)
            if debug: logger.warning(f"word '{word}' consists of tokens outside the dtw alignment range (0-{seq_len_att-1}), skipping timing")
            continue

        # find the start/end index for this word within the dtw-aligned tokens
        start_dtw_idx = min(dtw_indices_for_word)
        # end index is inclusive for the last token, so add 1 for slicing the boundaries array
        end_dtw_idx = max(dtw_indices_for_word) + 1

        # get corresponding frame boundaries from the dtw result
        start_frame = token_boundaries_frames[start_dtw_idx]
        # ensure end index doesn't go out of bounds for the boundaries array
        end_frame = token_boundaries_frames[end_dtw_idx] if end_dtw_idx < len(token_boundaries_frames) else n_frames_feature

        # convert frame boundaries to time relative to the start of the dtw alignment window
        start_time_rel = round(start_frame * AUDIO_TIME_PER_TOKEN, 3)
        end_time_rel = round(end_frame * AUDIO_TIME_PER_TOKEN, 3)

        # add the calculated anchor time to get the final timestamp relative to the segment start
        adjusted_start = round(anchor_time + start_time_rel, 3)
        adjusted_end = round(anchor_time + end_time_rel, 3)
        # ensure end time is not before start time after adjustments
        adjusted_end = max(adjusted_start, adjusted_end)

        word_timestamps_aligned.append({
            "word": word,
            "start": adjusted_start,
            "end": adjusted_end,
            "tokens": word_tokens_list[i], # store token ids for this word (mainly for debug)
            "token_indices": relative_indices_for_word # store original relative indices (mainly for debug/reconstruction)
        })
        # record the original indices (relative to generated_text_tokens_ids) covered by this word
        aligned_token_original_indices.update(relative_indices_for_word)

    # reconstruct output including potentially skipped tokens
    # the alignment process focuses on letter-containing tokens/words.
    # this step reconstructs the full sequence, adding back words/tokens that were skipped
    # (e.g., punctuation, words without letters) with None timestamps.
    output_with_skipped = []
    word_iter = iter(word_timestamps_aligned) # iterator over successfully aligned words
    current_word_entry = next(word_iter, None)

    # iterate through the *original* list of generated text tokens
    for idx, token_id in enumerate(generated_text_tokens_ids):
        token_str = tokenizer.decode([token_id]).strip()

        if idx in aligned_token_original_indices:
            # if this token index belongs to an aligned word, add the word entry
            # only add the word entry when encountering the *first* token of that word to avoid duplicates
            if current_word_entry and current_word_entry["token_indices"] and idx == current_word_entry["token_indices"][0]:
                output_with_skipped.append({
                    "word": current_word_entry["word"],
                    "start": current_word_entry["start"],
                    "end": current_word_entry["end"],
                })
                current_word_entry = next(word_iter, None) # advance to the next aligned word
        elif token_str and token_id < tokenizer.sot: # add skipped *non-special* tokens back as words with no timestamps
             output_with_skipped.append({
                 "word": token_str,
                 "start": None,
                 "end": None,
             })

    if not output_with_skipped and debug:
        logger.warning("dtw alignment resulted in an empty timestamp list after reconstruction")

    return output_with_skipped


def icu_tokenize_simple(text, lang='en'):
    """simple icu tokenizer for basic word splitting, used elsewhere potentially"""
    try:
        import icu
        from icu import Locale, BreakIterator
    except ImportError:
        # fallback to basic space splitting if icu not available
        return [word for word in text.split(' ') if word]
    try:
        locale = icu.Locale(lang)
        breaker = icu.BreakIterator.createWordInstance(locale)
    except Exception:
        # fallback if locale/breaker creation fails
        return [word for word in text.split(' ') if word]

    breaker.setText(text)
    tokens = []
    start = breaker.first()
    for end in breaker:
        token = text[start:end]
        if token.strip(): # only add non-empty tokens after stripping whitespace
            tokens.append(token.strip())
        start = end
    return tokens

def format_timestamp(seconds):
    """formats seconds into hh:mm:ss.ms string"""
    if seconds is None: return "n/a"
    milliseconds = round(seconds * 1000)
    ss = milliseconds // 1000
    ms = milliseconds % 1000
    mm = ss // 60
    ss %= 60
    hh = mm // 60
    mm %= 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"
