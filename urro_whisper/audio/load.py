import soundfile as sf
import numpy as np
import logging
import torch
import whisper # needed for log_mel_spectrogram

logger = logging.getLogger("urro_whisper") # use consistent logger name

try:
    import librosa
    librosa_available = True
except ImportError:
    librosa_available = False
    logger.info("librosa not installed, resampling will not be available if audio needs it")

TARGET_SAMPLE_RATE = 16000 # whisper standard sample rate

def load_audio_and_resample(audio_path, verbose=False):
    """loads audio, converts to mono float32, and resamples to target rate"""
    try:
        # load audio using soundfile
        audio_data_raw, sr_original = sf.read(audio_path, dtype='float32') # read directly as float32
        if verbose:
            logger.info(f"audio loaded: path='{audio_path}', original shape={audio_data_raw.shape}, original sample rate={sr_original}hz")
    except Exception as e:
        logger.error(f"error loading audio file '{audio_path}': {e}")
        raise # re-raise the exception for the caller to handle

    # ensure mono channel
    if audio_data_raw.ndim > 1:
        # average channels to convert to mono
        audio_mono = audio_data_raw.mean(axis=1)
        if verbose:
            logger.info(f"converted stereo/multi-channel audio to mono")
    else:
        audio_mono = audio_data_raw

    # resample if necessary
    if sr_original != TARGET_SAMPLE_RATE:
        if not librosa_available:
            logger.error(f"audio file '{audio_path}' has sample rate {sr_original}hz but target is {TARGET_SAMPLE_RATE}hz")
            logger.error("librosa is required for resampling but not installed, please install it (`pip install librosa`)")
            raise ImportError("librosa is required for resampling but not installed")

        if verbose:
            logger.info(f"resampling audio from {sr_original}hz to {TARGET_SAMPLE_RATE}hz using librosa")
        try:
            # librosa expects float input, which we already ensured
            audio_resampled = librosa.resample(y=audio_mono, orig_sr=sr_original, target_sr=TARGET_SAMPLE_RATE)
        except Exception as e:
            logger.error(f"error during librosa resampling: {e}")
            raise # re-raise resampling error
    else:
        # no resampling needed
        audio_resampled = audio_mono
        if verbose:
            logger.info("audio sample rate matches target, no resampling needed")

    # final audio is float32, 16khz, mono
    audio_final_processed = audio_resampled.astype(np.float32) # ensure correct dtype just in case
    if verbose:
        logger.info(f"audio processing complete: final shape={audio_final_processed.shape}, sample rate={TARGET_SAMPLE_RATE}hz")

    # return the processed audio data and its sample rate (which should always be TARGET_SAMPLE_RATE)
    return audio_final_processed, TARGET_SAMPLE_RATE

def calculate_mel_for_segment(audio_segment, model_name="tiny", n_mels_required=80, verbose=False):
    """calculates the log-mel spectrogram for an audio segment using whisper's method and pads/trims"""
    # whisper model expects input features for 30 seconds of audio
    TARGET_FRAMES_ENC = 3000 # = 30s * 16000hz / 160 hop_length / 2 ??? check this math

    # convert numpy segment to torch tensor
    audio_tensor = torch.from_numpy(audio_segment).float()

    if verbose:
        duration_s = len(audio_segment) / TARGET_SAMPLE_RATE
        logger.info(f"computing log-mel spectrogram ({n_mels_required} bins) for segment of length {len(audio_segment)} samples ({duration_s:.2f}s)")

    try:
        # use whisper's built-in function to compute log-mel spectrogram
        # this handles padding the audio internally to match hop_length requirements
        # n_mels should match the model being used (80 for most, 128 for large-v3)
        mel_spectrogram = whisper.log_mel_spectrogram(audio_tensor, n_mels=n_mels_required, padding=TARGET_FRAMES_ENC * whisper.audio.HOP_LENGTH) # pad audio first
        # trim to exact frame count *after* calculation
        mel_spectrogram = mel_spectrogram[:, :TARGET_FRAMES_ENC]

        # move to cpu and convert to numpy float32 for onnx inference
        mel_np = mel_spectrogram.cpu().numpy().astype(np.float32)

    except Exception as e:
        logger.error(f"error computing log-mel spectrogram: {e}")
        raise

    # add batch dimension expected by the onnx model
    if mel_np.ndim == 2:
        mel_np = mel_np[np.newaxis, :, :] # shape becomes (1, n_mels, n_frames)

    n_frames = mel_np.shape[-1]

    # double-check frame count and pad/trim if whisper's internal padding wasn't exact (shouldn't happen often with audio padding)
    if n_frames < TARGET_FRAMES_ENC:
        pad_width = TARGET_FRAMES_ENC - n_frames
        # pad the time dimension (last dimension) with zeros (constant mode)
        mel_np = np.pad(mel_np, ((0,0), (0,0), (0, pad_width)), mode='constant', constant_values=0)
        if verbose: logger.info(f"padded mel spectrogram frames by {pad_width} (target: {TARGET_FRAMES_ENC})")
    elif n_frames > TARGET_FRAMES_ENC:
        # trim the time dimension if it's somehow longer
        mel_np = mel_np[:, :, :TARGET_FRAMES_ENC]
        if verbose: logger.info(f"trimmed mel spectrogram frames from {n_frames} to {TARGET_FRAMES_ENC}")

    # validate final mel shape before returning
    expected_shape = (1, n_mels_required, TARGET_FRAMES_ENC)
    if mel_np.shape != expected_shape:
         # this would indicate a problem in the calculation or padding logic
         logger.error(f"final mel spectrogram shape is incorrect: {mel_np.shape}, expected {expected_shape}")
         raise ValueError(f"mel spectrogram final shape validation failed: got {mel_np.shape}, expected {expected_shape}")

    if verbose:
        logger.info(f"mel spectrogram calculation successful, final shape: {mel_np.shape}")

    return mel_np
