import onnxruntime as ort
import logging
from .download import download_model_files # ensure download functions are available

logger = logging.getLogger("urro_whisper") # use consistent logger name
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

def load_onnx_models(encoder_path, decoder_path, onnx_providers=None, verbose=False, exclude_providers=None):
    """loads onnx encoder and decoder models using onnxruntime, allowing provider selection/exclusion"""
    available_providers = ort.get_available_providers()
    logger.info(f"available onnx providers: {available_providers}")

    # determine providers to use
    if onnx_providers is not None:
        providers_to_use = onnx_providers # use user-specified providers if provided
    else:
        # default order attempts cuda/rocm first if available
        preferred_order = [
            'CUDAExecutionProvider',
            'ROCMExecutionProvider', # technically hip
            'CoreMLExecutionProvider',
            'CPUExecutionProvider',
        ]
        providers_to_use = [p for p in preferred_order if p in available_providers]
        if not providers_to_use: providers_to_use = available_providers # fallback to whatever ort found

    # filter out explicitly excluded providers
    if exclude_providers:
        original_providers = list(providers_to_use) # copy before modification
        providers_to_use = [p for p in providers_to_use if p not in exclude_providers]
        excluded_str = ", ".join(exclude_providers)
        if len(providers_to_use) < len(original_providers):
            logger.warning(f"excluding specified onnx providers: {excluded_str}")
        elif verbose:
            logger.info(f"providers {excluded_str} were specified for exclusion but not found in the available/selected list")

        if not providers_to_use and original_providers:
             logger.error("all specified or available onnx providers were excluded")
             # fallback to the original list before exclusion if exclusion resulted in empty list
             providers_to_use = original_providers
             logger.warning(f"falling back to providers before exclusion: {providers_to_use}")
        elif not providers_to_use:
             logger.error("no available providers remain after exclusion and no fallback possible")
             raise RuntimeError("no suitable onnx execution providers found after exclusion")

    if not providers_to_use:
        logger.error("no available onnx execution providers found or specified")
        raise RuntimeError("no available onnx execution providers")

    logger.info(f"attempting to load models using providers: {providers_to_use}")

    encoder_sess = None
    decoder_sess = None

    # load encoder
    if verbose: logger.info(f"loading onnx encoder: {encoder_path}")
    try:
        # configure session options if needed (e.g., logging level)
        sess_options = ort.SessionOptions()
        if verbose:
            sess_options.log_severity_level = 0 # detailed ort logging
        else:
            sess_options.log_severity_level = 3 # errors and warnings only

        encoder_sess = ort.InferenceSession(encoder_path, sess_options=sess_options, providers=providers_to_use)
    except Exception as e:
        logger.error(f"failed to load onnx encoder model with providers {providers_to_use}: {e}")
        # provide hint if external data file might be missing
        if ".onnx_data" in str(e): logger.error("hint: check if a corresponding '.onnx_data' file exists and is accessible next to the model")
        # note: you could implement more sophisticated fallback logic here, e.g., retry with cpu only
        raise

    # load decoder
    if verbose: logger.info(f"loading onnx decoder: {decoder_path}")
    try:
        # use the same session options as encoder for consistency
        sess_options = ort.SessionOptions()
        if verbose: sess_options.log_severity_level = 0
        else: sess_options.log_severity_level = 3

        decoder_sess = ort.InferenceSession(decoder_path, sess_options=sess_options, providers=providers_to_use)
    except Exception as e:
        logger.error(f"failed to load onnx decoder model with providers {providers_to_use}: {e}")
        if ".onnx_data" in str(e): logger.error("hint: check if a corresponding '.onnx_data' file exists and is accessible next to the model")
        # potential fallback logic here too
        raise

    if encoder_sess is None or decoder_sess is None:
        # this shouldn't happen if exceptions are raised correctly above, but as a safeguard
        raise RuntimeError("failed to initialize one or both onnx inference sessions")

    logger.info(f"onnx models loaded successfully using {encoder_sess.get_providers()} for encoder and {decoder_sess.get_providers()} for decoder")
    return encoder_sess, decoder_sess

def get_tokenizer_for_model(model, language, task):
    """loads the whisper tokenizer based on model name, language, and task"""
    from whisper.tokenizer import get_tokenizer # import here to avoid circular dependency if utils are restructured
    is_multilingual = not model.endswith(".en") # determine if model supports multiple languages
    try:
        tokenizer = get_tokenizer(multilingual=is_multilingual, task=task, language=language)
        logger.info(f"tokenizer loaded: multilingual={is_multilingual}, language={tokenizer.language}, task={task}")
        return tokenizer, is_multilingual
    except Exception as e:
         # provide a more informative error message
         raise RuntimeError(f"failed to load tokenizer for model '{model}' (multilingual={is_multilingual}, lang={language}, task={task}), error: {e}")
