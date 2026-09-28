import importlib.util
import unittest
from pathlib import Path


def _load_split():
    path = Path(__file__).resolve().parents[1] / "whisper_timestamped" / "make_subtitles.py"
    spec = importlib.util.spec_from_file_location("make_subtitles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.split_long_segments


split_long_segments = _load_split()


def _segment(words, text=None):
    return {
        "text": text if text is not None else "".join(word["text"] for word in words),
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": words,
    }


def _word(text, start, end):
    return {"text": text, "start": start, "end": end}


class SplitLongSegmentsTest(unittest.TestCase):
    def test_no_space_keeps_the_character_after_punctuation(self):
        segment = _segment([
            _word("ab.", 0.0, 1.0),
            _word("cd", 1.0, 2.0),
        ])
        got = split_long_segments([segment], max_length=3, use_space=False)
        self.assertEqual([part["text"] for part in got], ["ab.", "cd"])
        self.assertEqual(got[1]["start"], 1.0)

    def test_japanese_keeps_the_character_after_punctuation(self):
        segment = _segment([
            _word("こんにちは。", 0.0, 1.0),
            _word("元気", 1.0, 2.0),
        ])
        got = split_long_segments([segment], max_length=6, use_space=False)
        self.assertEqual([part["text"] for part in got], ["こんにちは。", "元気"])

    def test_no_space_multiple_cuts(self):
        segment = _segment([
            _word("ab。", 0.0, 1.0),
            _word("cd。", 1.0, 2.0),
            _word("ef", 2.0, 3.0),
        ])
        got = split_long_segments([segment], max_length=3, use_space=False)
        self.assertEqual([part["text"] for part in got], ["ab。", "cd。", "ef"])

    def test_space_still_splits_after_punctuation(self):
        segment = _segment(
            [
                _word("ab.", 0.0, 1.0),
                _word("cd", 1.0, 2.0),
            ],
            text="ab. cd",
        )
        got = split_long_segments([segment], max_length=4, use_space=True)
        self.assertEqual([part["text"] for part in got], ["ab.", "cd"])


if __name__ == "__main__":
    unittest.main()
