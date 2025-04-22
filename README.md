# Whisper with Urro `ᴡʜɪꜱᴘᴇʀ + ᴜʀʀᴏ`

Multilingual automatic speech recognition (ASR) with new speaker segmentation (NSS) and word-level timestamps (WLT) 

## Installation
```shell
pip install git+https://github.com/urroxyz/whisper.git
```

## Quickstart

```py
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from urro_whisper import whisperer, HYPHEN, GREATER, DOUBLE_GREATER

audio = "audio.wav" # Make sure this file exists or replace with a valid path
diarization_prefix = DOUBLE_GREATER

result = whisperer(
    model="tiny", # Use a valid model size like 'tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'
    audio=audio,
    language="en",
    diarization_prefix=diarization_prefix,
    verbose=False, # Set to True for detailed logging
)

print("\n--- Transcript ---")
# Access the transcript text using the simplified key "text"
texts = result["text"].split(diarization_prefix + " ")
for _, text in enumerate(texts):
    if text: # Avoid printing empty strings if splitting resulted in them
      print(text)

def format_timestamp(seconds):
    if seconds is None: return "N/A"
    milliseconds = round(seconds * 1000)
    ss = milliseconds // 1000
    ms = milliseconds % 1000
    mm = ss // 60
    ss %= 60
    hh = mm // 60
    mm %= 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

try:
    from IPython.display import display, HTML, Audio
    import soundfile as sf
    import math
    import numpy as np
    import librosa

    audio_original, sr_original = sf.read(audio)
    if audio_original.ndim > 1:
        audio_original = audio_original.mean(axis=1)

    target_sample_rate = 16000

    if sr_original != target_sample_rate:
        audio_playback = librosa.resample(
            y=audio_original.astype(np.float32),
            orig_sr=sr_original,
            target_sr=target_sample_rate
        )
    else:
        audio_playback = audio_original.astype(np.float32)

    html_rows = []
    html_rows.append("<tr><th>Timestamp</th><th>Text</th><th>Audio</th></tr>") # Changed Word -> Text

    # Access the word timestamps using the simplified key "words"
    # Access the word text using the simplified key "text" within each item
    for idx, word_info in enumerate(result["words"]):
        start_time = word_info['start']
        end_time = word_info['end']
        word_text = word_info['text'] # Changed 'word' -> 'text'
        ts_str = f"[{format_timestamp(start_time)} --> {format_timestamp(end_time)}]"
        audio_player_html = "N/A"
        if (
            start_time is not None
            and end_time is not None
            and end_time > start_time
        ):
            start_sample = max(0, math.floor(start_time * target_sample_rate))
            end_sample = min(len(audio_playback), math.ceil(end_time * target_sample_rate))

            if end_sample > start_sample:
                audio_segment = audio_playback[start_sample:end_sample]

                # Normalize segment for playback if needed (prevents clipping in Audio widget)
                max_abs = np.max(np.abs(audio_segment))
                if max_abs > 1.0:
                    audio_segment = audio_segment / max_abs
                elif max_abs == 0:
                     # Avoid division by zero for silent segments
                     pass # audio_segment is already all zeros

                try:
                    audio_obj = Audio(data=audio_segment, rate=target_sample_rate, autoplay=False)
                    audio_player_html = audio_obj._repr_html_()
                except Exception as audio_err:
                    print(f"Warning: Could not create audio player for segment '{word_text}': {audio_err}")
                    audio_player_html = "(Error creating player)"

            else:
                audio_player_html = "(empty segment)"
        html_rows.append(
            f"<tr><td>{ts_str}</td><td>{word_text}</td><td>{audio_player_html}</td></tr>"
        )
    html_table = (
        "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        "<thead></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
    )
    display(HTML(html_table))

except ImportError as e:
    print(f"\nSkipping HTML table generation due to missing libraries: {e}")
    print("You might need to install: pip install ipython soundfile librosa")
    print("\n--- Word-level Timestamps (Text Fallback) ---")
    # Fallback print using simplified names
    # Access the word timestamps using the simplified key "words"
    # Access the word text using the simplified key "text" within each item
    if "words" in result:
        for word_info in result["words"]:
            start = word_info['start']
            end = word_info['end']
            text_ = word_info['text'] # Changed 'word' -> 'text'
            print(f"[{format_timestamp(start)} --> {format_timestamp(end)}]\t{text_}")
    else:
        print("No word timestamp information available in results.")

except FileNotFoundError:
    print(f"\nError: Audio file not found at '{audio}'. Please provide a valid path.")
except Exception as e:
    print(f"\nAn error occurred during HTML table generation or fallback: {e}")
    import traceback
    traceback.print_exc()
```

## Acknowledgements
* [openai-whisper] by OpenAI
    * mel spectrogram handling
* [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) by Linto AI
    * extracting word-level timestamps
