import argparse
from whisper_timestamped.transcriber import WhisperTimestamped
import whisper

def main():
    parser = argparse.ArgumentParser(description="Run whisper-timestamped on an audio file.")
    parser.add_argument("--input", required=True, help="Path to audio file (e.g., .wav, .mp3)")
    parser.add_argument("--model", default="base", help="Whisper model to use (default: base)")
    parser.add_argument("--language", default=None, help="Force language (e.g., 'en')")
    args = parser.parse_args()

    model = whisper.load_model(args.model)
    result = model.transcribe(args.input, language=args.language)

    ts = WhisperTimestamped(result)
    timestamped = ts.get_timestamped_transcription()

    for word in timestamped:
        print(f"[{word['start']:.2f} - {word['end']:.2f}] {word['word']} (conf: {word['confidence']:.2f})")

if __name__ == "__main__":
    main()
