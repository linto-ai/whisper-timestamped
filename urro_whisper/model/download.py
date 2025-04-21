import os
import urllib.request
import logging
from tqdm import tqdm # for progress bar

logger = logging.getLogger("urro_whisper") # use consistent logger name

# define standard whisper model sizes known on hugging face community repos
STANDARD_SIZES = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo", # distil-large-v3 alias
]

# aliases for convenience
SIZE_ALIASES = {
    "large": "large-v3",
    "turbo": "large-v3-turbo",
}

# url templates for hugging face hub
HF_URL_TEMPLATE = "https://huggingface.co/{repo}/resolve/main/onnx/{file}?download=true"

def _download_file(url, dest, verbose=False, allow_404=False):
    """downloads a file from url to dest path, showing progress"""
    logger.info(f"downloading {url} to {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True) # ensure destination directory exists

    try:
        # use a user-agent to avoid potential blocking
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

        with urllib.request.urlopen(req) as response:
            # check http status code if needed, although urlopen raises HTTPError for >399
            if response.status != 200:
                 # this case might be rare if HTTPError is raised first
                 logger.warning(f"failed to download {url}: http status {response.status}")
                 return False

            total = int(response.info().get('Content-Length', -1)) # get file size for progress bar

            with open(dest, 'wb') as out_file, tqdm(
                total=total if total > 0 else None, # handle unknown size
                unit='B', unit_scale=True, unit_divisor=1024, # human-readable units
                desc=os.path.basename(dest), # show filename in progress bar
                ncols=80, # limit progress bar width
                disable=False # ensure progress bar is shown (not disabled by verbose flag)
            ) as bar:
                while True:
                    chunk = response.read(8192) # read in chunks
                    if not chunk:
                        break # end of file
                    out_file.write(chunk)
                    bar.update(len(chunk)) # update progress bar

        logger.info(f"downloaded {dest}")
        return True

    except urllib.error.HTTPError as e:
        if e.code == 404 and allow_404:
            logger.info(f"file not found (404) at {url} (allowed)")
            return False # allowed 404 is not an error, but download failed
        else:
            # log other http errors as warnings or errors
            logger.error(f"failed to download {url}: http error {e.code} {e.reason}")
            return False
    except Exception as e:
        logger.error(f"an unexpected error occurred downloading {url}: {e}")
        # clean up potentially incomplete file
        if os.path.exists(dest):
            try: os.remove(dest)
            except OSError: pass
        return False

def _try_download_data_file(repo, file_stem, model_dir, verbose=False):
    """attempts to download the associated .onnx_data file, allowing 404"""
    data_file_name = f"{file_stem}.onnx_data"
    data_file_path = os.path.join(model_dir, data_file_name)

    # skip download if file already exists
    if os.path.exists(data_file_path):
        if verbose: logger.info(f"data file {data_file_path} already exists, skipping download")
        return True

    url = HF_URL_TEMPLATE.format(repo=repo, file=data_file_name)
    logger.info(f"checking for optional data file: {url}")
    downloaded = _download_file(url, data_file_path, verbose=verbose, allow_404=True)

    if not downloaded and os.path.exists(data_file_path):
        # if download failed (e.g., 404) but left an empty file handle, remove it
        try: os.remove(data_file_path)
        except OSError: pass

    # return value indicates if file exists *after* the attempt (true if existed before or downloaded)
    return os.path.exists(data_file_path)


def download_model_files(model_identifier, model_dir, verbose=False):
    """
    downloads whisper onnx model files (encoder, decoder, and potential data files)
    for a given model identifier (e.g., 'base.en', 'large-v3', or custom repo 'user/my-whisper').
    """
    os.makedirs(model_dir, exist_ok=True)
    model_key = model_identifier.lower()
    model_size = SIZE_ALIASES.get(model_key, model_key) # resolve aliases

    # determine primary repo and potential fallback for standard sizes
    is_standard = model_size in STANDARD_SIZES
    if is_standard:
        # prefer timestamped versions if available
        repo = f"onnx-community/whisper-{model_size}_timestamped"
        fallback_repo = f"onnx-community/whisper-{model_size}-ONNX"
    else:
        # assume model_identifier is a custom hugging face repo path
        repo = model_identifier
        fallback_repo = None
        logger.info(f"attempting download from custom repo: {repo}")

    # define expected file names
    encoder_file = os.path.join(model_dir, "encoder_model.onnx")
    decoder_file = os.path.join(model_dir, "decoder_model.onnx")
    # some repos might have a merged decoder (weights included)
    decoder_merged_file = os.path.join(model_dir, "decoder_model_merged.onnx")

    # download encoder
    if not os.path.exists(encoder_file):
        logger.info("encoder model not found locally, attempting download")
        encoder_url = HF_URL_TEMPLATE.format(repo=repo, file="encoder_model.onnx")
        ok = _download_file(encoder_url, encoder_file, verbose=verbose, allow_404=(fallback_repo is not None))

        if not ok and fallback_repo:
            logger.info(f"encoder download from {repo} failed or not found, trying fallback {fallback_repo}")
            fallback_encoder_url = HF_URL_TEMPLATE.format(repo=fallback_repo, file="encoder_model.onnx")
            ok = _download_file(fallback_encoder_url, encoder_file, verbose=verbose)

        if not ok: # if still not downloaded after primary and fallback attempts
            logger.error(f"failed to download encoder model for '{model_identifier}' from primary repo '{repo}'" + (f" and fallback repo '{fallback_repo}'" if fallback_repo else ""))
            raise FileNotFoundError(f"could not download encoder for model '{model_identifier}'")
    else:
        if verbose: logger.info(f"encoder model found locally: {encoder_file}")

    # download decoder
    # check if either standard or merged decoder exists locally
    decoder_exists = os.path.exists(decoder_file)
    merged_decoder_exists = os.path.exists(decoder_merged_file)

    if not (decoder_exists or merged_decoder_exists):
        logger.info("decoder model (standard or merged) not found locally, attempting download")
        # try downloading the standard (split weights) decoder first
        decoder_url = HF_URL_TEMPLATE.format(repo=repo, file="decoder_model.onnx")
        ok = _download_file(decoder_url, decoder_file, verbose=verbose, allow_404=True) # allow 404 if merged exists

        # if standard decoder download failed or wasn't found, try the merged one
        if not ok:
            logger.info("standard decoder download failed or not found, trying merged decoder")
            decoder_merged_url = HF_URL_TEMPLATE.format(repo=repo, file="decoder_model_merged.onnx")
            ok = _download_file(decoder_merged_url, decoder_merged_file, verbose=verbose, allow_404=(fallback_repo is not None))

        # if still not ok after trying both from primary repo, try fallback repo
        if not ok and fallback_repo:
            logger.info(f"decoder download from {repo} failed or not found, trying fallback {fallback_repo}")
            # try fallback standard decoder
            fallback_decoder_url = HF_URL_TEMPLATE.format(repo=fallback_repo, file="decoder_model.onnx")
            ok = _download_file(fallback_decoder_url, decoder_file, verbose=verbose, allow_404=True)
            # try fallback merged decoder if standard failed
            if not ok:
                logger.info("fallback standard decoder download failed or not found, trying fallback merged decoder")
                fallback_decoder_merged_url = HF_URL_TEMPLATE.format(repo=fallback_repo, file="decoder_model_merged.onnx")
                ok = _download_file(fallback_decoder_merged_url, decoder_merged_file, verbose=verbose)

        # check existence again after all attempts
        decoder_exists = os.path.exists(decoder_file)
        merged_decoder_exists = os.path.exists(decoder_merged_file)
        if not (decoder_exists or merged_decoder_exists):
            logger.error(f"failed to download decoder model for '{model_identifier}' from primary repo '{repo}'" + (f" and fallback repo '{fallback_repo}'" if fallback_repo else ""))
            raise FileNotFoundError(f"could not download decoder for model '{model_identifier}'")
    else:
         if verbose:
             if merged_decoder_exists: logger.info(f"merged decoder model found locally: {decoder_merged_file}")
             elif decoder_exists: logger.info(f"standard decoder model found locally: {decoder_file}")


    # download data files (optional)
    # try downloading data files for both primary and fallback repos, allowing 404
    if verbose: logger.info("checking for optional .onnx_data files")
    _try_download_data_file(repo, "encoder_model", model_dir, verbose=verbose)
    # decide which decoder data file to check based on which decoder file exists
    if merged_decoder_exists:
        _try_download_data_file(repo, "decoder_model_merged", model_dir, verbose=verbose)
    elif decoder_exists:
        _try_download_data_file(repo, "decoder_model", model_dir, verbose=verbose)

    if fallback_repo:
        if verbose: logger.info(f"checking for optional .onnx_data files in fallback repo {fallback_repo}")
        _try_download_data_file(fallback_repo, "encoder_model", model_dir, verbose=verbose)
        if merged_decoder_exists:
             _try_download_data_file(fallback_repo, "decoder_model_merged", model_dir, verbose=verbose)
        elif decoder_exists:
             _try_download_data_file(fallback_repo, "decoder_model", model_dir, verbose=verbose)

    # final check
    encoder_ok = os.path.exists(encoder_file)
    decoder_ok = os.path.exists(decoder_file) or os.path.exists(decoder_merged_file)
    if not (encoder_ok and decoder_ok):
        logger.error(f"critical error: after download attempts, required model files are still missing for '{model_identifier}' in {model_dir}")
        raise FileNotFoundError(f"could not obtain all required model files for '{model_identifier}'")

    if verbose:
        logger.info(f"model files for '{model_identifier}' verified in {model_dir}")


def get_onnx_model_paths(
    model_identifier, # e.g., 'base.en', 'large-v3', 'user/my-model'
    onnx_encoder_path=None, # user override
    onnx_decoder_path=None, # user override
    verbose=False,
    cache_dir=None # optional override for cache location
):
    """
    resolves paths to onnx model files. uses user-provided paths if valid,
    otherwise checks cache or downloads models.
    """
    # check if user provided valid paths for both encoder and decoder
    if onnx_encoder_path and os.path.isfile(onnx_encoder_path) and \
       onnx_decoder_path and os.path.isfile(onnx_decoder_path):
        logger.info(f"using user-provided onnx paths: encoder='{onnx_encoder_path}', decoder='{onnx_decoder_path}'")
        # check for associated data files if user provided paths
        enc_data = onnx_encoder_path + "_data"
        dec_data = onnx_decoder_path + "_data"
        if not os.path.exists(enc_data): logger.warning(f"user-provided encoder might be missing data file: {enc_data}")
        if not os.path.exists(dec_data): logger.warning(f"user-provided decoder might be missing data file: {dec_data}")
        return onnx_encoder_path, onnx_decoder_path

    # determine cache directory
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "urro_whisper", "models")
    logger.info(f"using model cache directory: {cache_dir}")

    # normalize model identifier and handle aliases
    model_key = model_identifier.lower()
    model_size_or_repo = SIZE_ALIASES.get(model_key, model_key)

    # construct expected directory and file paths within the cache
    # replace slashes in repo names for safe directory creation
    model_dir_name = model_size_or_repo.replace("/", "--")
    model_dir = os.path.join(cache_dir, model_dir_name)

    encoder_path = os.path.join(model_dir, "encoder_model.onnx")
    decoder_path = os.path.join(model_dir, "decoder_model.onnx")
    decoder_merged_path = os.path.join(model_dir, "decoder_model_merged.onnx")

    # check if required files exist in cache, download if not
    # prefer merged decoder if it exists
    final_decoder_path = decoder_merged_path if os.path.exists(decoder_merged_path) else decoder_path

    if not os.path.exists(encoder_path) or not os.path.exists(final_decoder_path):
        logger.info(f"model '{model_identifier}' not found in cache, initiating download to {model_dir}")
        try:
            # pass the original identifier (or alias resolution) to the download function
            download_model_files(model_size_or_repo, model_dir, verbose=verbose)
            # re-determine final decoder path after download
            final_decoder_path = decoder_merged_path if os.path.exists(decoder_merged_path) else decoder_path
            # final check after download attempt
            if not os.path.exists(encoder_path) or not os.path.exists(final_decoder_path):
                 raise FileNotFoundError("model files still missing after download attempt")
        except Exception as e:
             logger.error(f"failed to download model '{model_identifier}': {e}")
             raise # re-raise the exception

    logger.info(f"using cached model files: encoder='{encoder_path}', decoder='{final_decoder_path}'")
    return encoder_path, final_decoder_path
