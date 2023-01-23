__author__ = "Jérôme Louradour"
__credits__ = ["Jérôme Louradour"]
__license__ = "MIT"

import unittest
import sys
import os
import subprocess
import shutil
import tempfile
import json
import torch

FAIL_IF_REFERENCE_NOT_FOUND = True
GENERATE_NEW_ONLY = False
GENERATE_ALL = False
GENERATE_DEVICE_DEPENDENT = False


class TestHelper(unittest.TestCase):

    def skipLongTests(self):
        return not torch.cuda.is_available()

    def setUp(self):
        self.maxDiff = None
        self.createdReferences = []

    def tearDown(self):
        if GENERATE_ALL or GENERATE_NEW_ONLY or not FAIL_IF_REFERENCE_NOT_FOUND or GENERATE_DEVICE_DEPENDENT:
            if len(self.createdReferences) > 0:
                print("WARNING: Created references: " +
                      ", ".join(self.createdReferences).replace(self.get_data_path()+"/", ""))
        else:
            self.assertEqual(self.createdReferences, [], "Created references: " +
                             ", ".join(self.createdReferences).replace(self.get_data_path()+"/", ""))

    def get_main_path(self, fn=None, check=False):
        return self._get_path("whisper_timestamped", fn, check=check)

    def get_output_path(self, fn=None):
        if fn == None:
            return tempfile.gettempdir()
        return os.path.join(tempfile.gettempdir(), fn)

    def get_expected_path(self, fn=None, check=False):
        return self._get_path("tests/expected", fn, check=check)

    def get_data_files(self, files=None, excluded_by_default=["apollo11.mp3", "music.mp4", "arabic.mp3"]):
        if files == None:
            files = os.listdir(self.get_data_path())
            files = [f for f in files if f not in excluded_by_default]
            files = sorted(files)
        return [self.get_data_path(fn) for fn in files]

    def get_generated_files(self, input_filename, output_path, extensions):
        for ext in extensions:
            yield os.path.join(output_path, os.path.basename(input_filename) + "." + ext.lstrip("."))

    def assertRun(self, cmd):
        if isinstance(cmd, str):
            return self.assertRun(cmd.split())
        curdir = os.getcwd()
        os.chdir(tempfile.gettempdir())
        if cmd[0].endswith(".py"):
            cmd = [sys.executable] + cmd
        print("Running:", " ".join(cmd))
        p = subprocess.Popen(cmd,
                             # Otherwise ".local" path might be missing
                             env=dict(
                                 os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE
                             )
        os.chdir(curdir)
        (stdout, stderr) = p.communicate()
        self.assertEqual(p.returncode, 0, msg=stderr.decode("utf-8"))
        return (stdout.decode("utf-8"), stderr.decode("utf-8"))

    def assertNonRegression(self, content, reference):
        """
        Check that a file/folder is the same as a reference file/folder.
        """
        if isinstance(content, dict):
            # Make a temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
                content = f.name
            res = self.assertNonRegression(f.name, reference)
            os.remove(f.name)
            return res
        elif not isinstance(content, str):
            raise ValueError(f"Invalid content type: {type(content)}")

        self.assertTrue(os.path.exists(content), f"Missing file: {content}")
        is_file = os.path.isfile(reference) if os.path.exists(
            reference) else os.path.isfile(content)

        reference = self.get_expected_path(
            reference, check=FAIL_IF_REFERENCE_NOT_FOUND)
        if not os.path.exists(reference) or ((GENERATE_ALL or GENERATE_DEVICE_DEPENDENT) and reference not in self.createdReferences):
            dirname = os.path.dirname(reference)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            if is_file:
                shutil.copyfile(content, reference)
            else:
                shutil.copytree(content, reference)
            self.createdReferences.append(reference)

        if is_file:
            self.assertTrue(os.path.isfile(content))
            self._check_file_non_regression(content, reference)
        else:
            self.assertTrue(os.path.isdir(content))
            for root, dirs, files in os.walk(content):
                for f in files:
                    f_ref = os.path.join(reference, f)
                    self.assertTrue(os.path.isfile(f_ref),
                                    f"Additional file: {f}")
                    self._check_file_non_regression(
                        os.path.join(root, f), f_ref)
            for root, dirs, files in os.walk(reference):
                for f in files:
                    f = os.path.join(content, f)
                    self.assertTrue(os.path.isfile(f), f"Missing file: {f}")

    def get_data_path(self, fn=None, check=True):
        return self._get_path("tests/data", fn, check)

    def _get_path(self, prefix, fn=None, check=True):
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            prefix
        )
        if fn:
            path = os.path.join(path, fn)
        if check:
            self.assertTrue(os.path.exists(path), f"Cannot find {path}")
        return path

    def _check_file_non_regression(self, file, reference):
        if file.endswith(".json"):
            with open(file) as f:
                content = json.load(f)
            with open(reference) as f:
                reference_content = json.load(f)
            self.assertClose(content, reference_content,
                             msg=f"File {file} does not match reference {reference}")
            return
        with open(file) as f:
            content = f.readlines()
        with open(reference) as f:
            reference_content = f.readlines()
        self.assertEqual(content, reference_content,
                         msg=f"File {file} does not match reference {reference}")

    def assertClose(self, obj1, obj2, msg=None):
        return self.assertEqual(self.loose(obj1), self.loose(obj2), msg=msg)

    def loose(self, obj):
        # Return an approximative value of an object
        if isinstance(obj, list):
            return [self.loose(a) for a in obj]
        if isinstance(obj, float):
            f = round(obj, 1)
            return 0.0 if f == -0.0 else f
        if isinstance(obj, dict):
            return {k: self.loose(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(self.loose(list(obj)))
        if isinstance(obj, set):
            return self.loose(list(obj), "set")
        return obj

    def get_audio_duration(self, audio_file):
        # Get the duration in sec *without introducing additional dependencies*
        import whisper
        return len(whisper.load_audio(audio_file)) / whisper.audio.SAMPLE_RATE

    def get_device_str(self):
        import torch
        return "cpu" if not torch.cuda.is_available() else "cuda"


class TestHelperCli(TestHelper):

    def _test_cli_(self, opts, name, files=None, extensions=["words.json"], prefix=None):

        output_dir = self.get_output_path(name)

        for input_filename in self.get_data_files(files):

            # Butterfly effect: Results are different depending on the device for long files
            duration = self.get_audio_duration(input_filename)
            device_dependent = duration > 60 or (
                duration > 30 and "tiny_fr" in name)
            name_ = name
            if device_dependent:
                name_ += f".{self.get_device_str()}"

            def ref_name(output_filename):
                return name_ + "/" + (f"{prefix}_" if prefix else "") + os.path.basename(output_filename)
            generic_name = ref_name(input_filename + ".*")

            if GENERATE_DEVICE_DEPENDENT and not name_.endswith("."+self.get_device_str()):
                print("Skipping non-regression test", generic_name)
                continue

            if GENERATE_NEW_ONLY and min([os.path.exists(self.get_expected_path(ref_name(output_filename)))
                                          for output_filename in self.get_generated_files(input_filename, output_dir, extensions=extensions)]
                                         ):
                print("Skipping non-regression test", generic_name)
                continue

            print("Running non-regression test", generic_name)

            main_script = self.get_main_path("transcribe.py", check=False)
            if not os.path.exists(main_script):
                main_script = "whisper_timestamped"
            (stdout, stderr) = self.assertRun([
                main_script,
                input_filename,
                "--output_dir", output_dir,
                *opts,
            ])
            print(stdout)
            print(stderr)
            for output_filename in self.get_generated_files(input_filename, output_dir, extensions=extensions):
                self.assertNonRegression(
                    output_filename, ref_name(output_filename))

        shutil.rmtree(output_dir, ignore_errors=True)


class TestTranscribeTiny(TestHelperCli):

    def test_cli_tiny_auto(self):
        self._test_cli_(
            ["--model", "tiny"],
            "tiny_auto",
        )

    def test_cli_tiny_fr(self):
        self._test_cli_(
            ["--model", "tiny", "--language", "fr"],
            "tiny_fr",
        )


class TestTranscribeMedium(TestHelperCli):

    def test_cli_medium_auto(self):
        self._test_cli_(
            ["--model", "medium"],
            "medium_auto",
        )

    def test_cli_medium_fr(self):
        self._test_cli_(
            ["--model", "medium", "--language", "fr"],
            "medium_fr",
        )


class TestTranscribeNaive(TestHelperCli):

    def test_naive(self):

        self._test_cli_(
            ["--model", "small", "--language", "en", "--naive"],
            "naive",
            files=["apollo11.mp3"],
            prefix="naive",
        )

        self._test_cli_(
            ["--model", "small", "--language", "en", "--accurate"],
            "naive",
            files=["apollo11.mp3"],
            prefix="accurate",
        )

    def test_stucked_segments(self):
        self._test_cli_(
            ["--model", "tiny", "--accurate"],
            "corner_cases",
            files=["apollo11.mp3"],
            prefix="accurate.tiny",
        )


class TestTranscribeCornerCases(TestHelperCli):

    def test_stucked_lm(self):
        if self.skipLongTests():
            return

        self._test_cli_(
            ["--model", "small", "--language", "en"],
            "corner_cases",
            files=["apollo11.mp3"],
            prefix="stucked_lm",
        )

    def test_temperature(self):

        self._test_cli_(
            ["--model", "small", "--language", "English",
                "--condition", "False", "--temperature", "0.1"],
            "corner_cases",
            files=["apollo11.mp3"],
            prefix="random.nocond",
        )

        if self.skipLongTests():
            return

        self._test_cli_(
            ["--model", "small", "--language", "en", "--temperature", "0.2"],
            "corner_cases",
            files=["apollo11.mp3"],
            prefix="random",
        )

    def test_not_conditioned(self):
        if not os.path.exists(self.get_data_path("music.mp4", check=False)):
            return
        if self.skipLongTests():
            return

        self._test_cli_(
            ["--model", "medium", "--language", "en", "--condition", "False"],
            "corner_cases",
            files=["music.mp4"],
            prefix="nocond",
        )

        self._test_cli_(
            ["--model", "medium", "--language", "en",
                "--condition", "False", "--temperature", "0.4"],
            "corner_cases",
            files=["music.mp4"],
            prefix="nocond.random",
        )

    def test_large(self):
        if self.skipLongTests():
            return

        self._test_cli_(
            ["--model", "large-v2", "--language", "en",
                "--condition", "False", "--temperature", "0.4"],
            "corner_cases",
            files=["apollo11.mp3"],
            prefix="large",
        )

        if os.path.exists(self.get_data_path("arabic.mp3", check=False)):
            self._test_cli_(
                ["--model", "large-v2", "--language", "Arabic"],
                "corner_cases",
                files=["arabic.mp3"]
            )


class TestTranscribeFormats(TestHelperCli):

    def test_cli_punctuations(self):
        files = ["punctuations.mp3"]
        extensions = ["txt", "srt", "vtt", "words.srt", "words.vtt",
                      "words.json", "csv", "words.csv", "tsv", "words.tsv"]
        opts = ["--model", "medium", "--language", "fr"]

        # An audio / model combination that produces coma
        self._test_cli_(
            opts,
            "punctuations_yes",
            files=files,
            extensions=extensions
        )
        self._test_cli_(
            opts + ["--punctuations", "False"],
            "punctuations_no",
            files=files,
            extensions=extensions
        )


# "ZZZ" to run this test at last (because it will fill the CUDA with some memory)
class TestZZZPythonImport(TestHelper):

    def test_python_import(self):

        try:
            import whisper_timestamped
        except ModuleNotFoundError:
            sys.path.append(os.path.realpath(
                os.path.dirname(os.path.dirname(__file__))))
            import whisper_timestamped

        self.assertTrue(isinstance(whisper_timestamped.__version__, str))

        model = whisper_timestamped.load_model("tiny")

        for filename in "bonjour.wav", "laugh1.mp3", "laugh2.mp3":
            res = whisper_timestamped.transcribe(
                model, self.get_data_path(filename))
            self.assertNonRegression(res, f"tiny_auto/{filename}.words.json")

        for filename in "bonjour.wav", "laugh1.mp3", "laugh2.mp3":
            res = whisper_timestamped.transcribe(
                model, self.get_data_path(filename), language="fr")
            self.assertNonRegression(res, f"tiny_fr/{filename}.words.json")

    def test_split_tokens(self):

        import whisper_timestamped as whisper
        from whisper_timestamped.transcribe import split_tokens_on_spaces

        tokenizer = whisper.tokenizer.get_tokenizer(True, language=None)

        # 220 means space
        tokens = [50364, 220, 6455, 11, 2232, 11, 286, 2041, 11, 2232, 11, 8660,
                  291, 808, 493, 220, 365, 11, 220, 445, 718, 505, 458, 13, 220, 50714]

        self.assertEqual(
            split_tokens_on_spaces(tokens, tokenizer),
            (['<|0.00|>', 'So,', 'uh,', 'I', 'guess,', 'uh,', 'wherever', 'you', 'come', 'up', 'with,', 'just', 'let', 'us', 'know.', '<|7.00|>'],
             [[50364],
                [220, 6455, 11],
                [2232, 11], [286], [2041, 11], [
                    2232, 11], [8660], [291], [808],
                [493, 220],
                [365, 11, 220],
                [445], [718], [505],
                [458, 13, 220],
                [50714]
              ])
        )

    def get_expected(self, input, dirname):
        with open(self.get_expected_path(f"{dirname}/{input}.words.json", check=True)) as f:
            return json.load(f)
