#!/usr/bin/env python3

import json
import string

_punctuation = "".join(c for c in string.punctuation if c not in ["-", "'"]) + "。，！？：”、…"

def split_long_segments(segments, max_length, use_space = True):
    new_segments = []
    for segment in segments:
        text = segment["text"]
        if len(text) <= max_length:
            new_segments.append(segment)
        else:
            meta_words = segment["words"]
            # Note: we do this in case punctuation were removed from words
            if use_space:
                # Split text around spaces and punctuations (keeping punctuations)
                words = text.split()
            else:
                words = [w["text"] for w in meta_words]
            if len(words) != len(meta_words):
                new_words = [w["text"] for w in meta_words]
                print(f"WARNING: {' '.join(words)} != {' '.join(new_words)}")
                words = new_words
            current_text = ""
            current_start = segment["start"]
            current_best_idx = None
            current_best_end = None
            current_best_next_start = None
            for i, (word, meta) in enumerate(zip(words, meta_words)):
                current_text_before = current_text
                if current_text and use_space:
                    current_text += " "
                current_text += word

                if len(current_text) > max_length and len(current_text_before):
                    start = current_start
                    if current_best_idx is not None:
                        text = current_text[:current_best_idx]
                        end = current_best_end
                        current_text = current_text[current_best_idx+1:]
                        current_start = current_best_next_start
                    else:
                        text = current_text_before
                        end = meta_words[i-1]["end"]
                        current_text = word
                        current_start = meta["start"]

                    current_best_idx = None
                    current_best_end = None
                    current_best_next_start = None                        

                    new_segments.append({"text": text, "start": start, "end": end})

                # Try to cut after punctuation
                if current_text and current_text[-1] in _punctuation:
                    current_best_idx = len(current_text)
                    current_best_end = meta["end"]
                    current_best_next_start = meta_words[i+1]["start"] if i+1 < len(meta_words) else None
            
            if len(current_text):
                new_segments.append({"text": current_text, "start": current_start, "end": segment["end"]})
            
    return new_segments

def cli():

    import os
    import argparse

    supported_formats = ["srt", "vtt"]

    parser = argparse.ArgumentParser(
        description='Convert .word.json transcription files (output of whisper_timestamped) to srt or vtt, being able to cut long segments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', type=str, help='Input json file, or input folder')
    parser.add_argument('output', type=str, help='Output srt or vtt file, or output folder')
    parser.add_argument('--max_length', default=200, help='Maximum length of a segment in characters', type=int)
    parser.add_argument('--format', type=str, default="all", help='Output format (if the output is a folder, i.e. not a file with an explicit extension)', choices= supported_formats + ["all"])
    args = parser.parse_args()

    if os.path.isdir(args.input) or not max([args.output.endswith(e) for e in supported_formats]):
        input_files = [f for f in os.listdir(args.input) if f.endswith(".words.json")] if os.path.isdir(args.input) else [os.path.basename(args.input)]
        extensions = [args.format] if args.format != "all" else ["srt", "vtt"]
        output_files = [[os.path.join(args.output, f[:-11] + "." + e) for e in extensions] for f in input_files]
        if os.path.isdir(args.input):
            input_files = [os.path.join(args.input, f) for f in input_files]
        else:
            input_files = [args.input]
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
    else:
        input_files = [args.input]
        output_files = [[args.output]]
        if not os.path.isdir(os.path.dirname(args.output)):
            os.makedirs(os.path.dirname(args.output))

    try:
        # Old whisper version # Before https://github.com/openai/whisper/commit/da600abd2b296a5450770b872c3765d0a5a5c769
        from whisper.utils import write_txt, write_srt, write_vtt

    except ImportError:
        # New whisper version
        from whisper.utils import get_writer

        def do_write(transcript, file, format):
            writer = get_writer(format, os.path.curdir)
            return writer.write_result({"segments": transcript}, file)
        def get_do_write(format):
            return lambda transcript, file: do_write(transcript, file, format)

        write_srt = get_do_write("srt")
        write_vtt = get_do_write("vtt")

    for fn, outputs in zip(input_files, output_files):
        with open(fn, "r") as f:
            transcript = json.load(f)
        segments = transcript["segments"]
        if args.max_length:
            language = transcript["language"]
            use_space = language not in ["zh", "ja", "th", "lo", "my"]
            segments = split_long_segments(segments, args.max_length, use_space=use_space)
        for output in outputs:
            if output.endswith(".srt"):
                with open(output, "w") as f:
                    write_srt(segments, file=f)
            elif output.endswith(".vtt"):
                with open(output, "w") as f:
                    write_vtt(segments, file=f)
            else:
                raise RuntimeError(f"Unknown output format for {output}")

if __name__ == "__main__":
    cli()