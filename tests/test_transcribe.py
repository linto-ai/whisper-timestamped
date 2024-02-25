__author__ = "Jérôme Louradour"
__credits__ = ["Jérôme Louradour"]
__license__ = "GPLv3"

import unittest
import sys
import os
import subprocess
import shutil
import tempfile
import json
import torch
import jsonschema

FAIL_IF_REFERENCE_NOT_FOUND = True
GENERATE_NEW_ONLY = False
GENERATE_ALL = False
GENERATE_DEVICE_DEPENDENT = False
SKIP_LONG_TEST_IF_CPU = True
CMD_OPTIONS = []


class TestHelper(unittest.TestCase):

    def skipLongTests(self):
        return SKIP_LONG_TEST_IF_CPU and not torch.cuda.is_available()

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
        return os.path.join(tempfile.gettempdir(), fn + self._extra_cmd_options())

    def get_expected_path(self, fn=None, check=False):
        return self._get_path("tests/expected" + self._extra_cmd_options(), fn, check=check)
    
    def _extra_cmd_options(self):
        s = "".join([f.replace("-","").strip() for f in CMD_OPTIONS])
        if s:
            return "." + s
        return ""

    def get_data_files(self, files=None, excluded_by_default=["apollo11.mp3", "music.mp4", "arabic.mp3", "japanese.mp3", "empty.wav", "words.wav"]):
        if files == None:
            files = os.listdir(self.get_data_path())
            files = [f for f in files if f not in excluded_by_default and not f.endswith("json")]
            files = sorted(files)
        return [self.get_data_path(fn) for fn in files]

    def get_generated_files(self, input_filename, output_path, extensions):
        for ext in extensions:
            yield os.path.join(output_path, os.path.basename(input_filename) + "." + ext.lstrip("."))

    def main_script(self, pyscript = "transcribe.py", exename = "whisper_timestamped"):
        main_script = self.get_main_path(pyscript, check=False)
        if not os.path.exists(main_script):
            main_script = exename
        return main_script

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

    def assertNonRegression(self, content, reference, string_is_file=True):
        """
        Check that a file/folder is the same as a reference file/folder.
        """
        if isinstance(content, dict):
            # Make a temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf8", delete=False) as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
                content = f.name
            res = self.assertNonRegression(f.name, reference)
            os.remove(f.name)
            return res
        elif not isinstance(content, str):
            raise ValueError(f"Invalid content type: {type(content)}")

        if not string_is_file:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf8", delete=False) as f:
                f.write(content)
                content = f.name 
            res = self.assertNonRegression(f.name, reference)
            os.remove(f.name)
            return res

        self.assertTrue(os.path.exists(content), f"Missing file: {content}")
        is_file = os.path.isfile(reference) if os.path.exists(reference) else os.path.isfile(content)

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
            if "language" in content and "language" in reference_content:
                content["language"] = self.norm_language(content["language"])
                reference_content["language"] = self.norm_language(reference_content["language"])
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

    def norm_language(self, language):
        # Cheap custom stuff to avoid importing everything
        return {
            "japanese": "ja",
        }.get(language.lower(), language)


class TestHelperCli(TestHelper):

    json_schema = None

    def _test_cli_(self, opts, name, files=None, extensions=["words.json"], prefix=None, one_per_call=True, device_specific=None):
        """
        Test command line
        opts: list of options
        name: name of the test
        files: list of files to process
        extensions: list of extensions to check, or None to test the stdout
        prefix: prefix to add to the reference files
        one_per_call: if True, each file is processed separately, otherwise all files are processed by a single process
        """

        opts = opts + CMD_OPTIONS

        output_dir = self.get_output_path(name)

        input_filenames = self.get_data_files(files)

        for i, input_filename in enumerate(input_filenames):

            # Butterfly effect: Results are different depending on the device for long files
            duration = self.get_audio_duration(input_filename)
            if device_specific is None:
                device_dependent = duration > 60 or (duration > 30 and "tiny_fr" in name) or ("empty" in input_filename and "medium_auto" in name)
            else:
                device_dependent = device_specific
            name_ = name
            if device_dependent and self.get_device_str() != "cuda":
                name_ += f".{self.get_device_str()}"

            def ref_name(output_filename):
                return name_ + "/" + (f"{prefix}_" if prefix else "") + os.path.basename(output_filename)
            generic_name = ref_name(input_filename + ".*")

            if GENERATE_DEVICE_DEPENDENT and not device_dependent:
                print("Skipping non-regression test", generic_name)
                continue

            if GENERATE_NEW_ONLY and min([os.path.exists(self.get_expected_path(ref_name(output_filename)))
                                          for output_filename in self.get_generated_files(input_filename, output_dir, extensions=extensions)]
                                         ):
                print("Skipping non-regression test", generic_name)
                continue

            print("Running non-regression test", generic_name)

            if one_per_call or i == 0:
                if one_per_call:
                    (stdout, stderr) = self.assertRun([self.main_script(), input_filename, "--output_dir", output_dir, *opts])
                else:
                    (stdout, stderr) = self.assertRun([self.main_script(), *input_filenames, "--output_dir", output_dir, *opts])
                print(stdout)
                print(stderr)

            output_json = self.get_generated_files(input_filename, output_dir, extensions=["words.json"]).__next__()
            if os.path.isfile(output_json):
                self.check_json(output_json)

            if extensions is None:
                output_filename = list(self.get_generated_files(input_filename, output_dir, extensions=["stdout"]))[0]
                self.assertNonRegression(stdout, ref_name(output_filename), string_is_file=False)
            else:
                for output_filename in self.get_generated_files(input_filename, output_dir, extensions=extensions):
                    self.assertNonRegression(output_filename, ref_name(output_filename))


        shutil.rmtree(output_dir, ignore_errors=True)

    def check_json(self, json_file):
        with open(json_file) as f:
            content = json.load(f)

        if self.json_schema is None:
            schema_file = os.path.join(os.path.dirname(__file__), "json_schema.json")
            self.assertTrue(os.path.isfile(schema_file), msg=f"Schema file {schema_file} not found")
            self.json_schema = json.load(open(schema_file))

        jsonschema.validate(instance=content, schema=self.json_schema)



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
            ["--model", "small", "--language", "en", "--efficient", "--naive"],
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
            ["--model", "tiny"],
            "corner_cases",
            files=["apollo11.mp3"],
            prefix="accurate.tiny",
        )


class TestTranscribeCornerCases(TestHelperCli):

    def test_stucked_lm(self):
        if self.skipLongTests():
            return

        self._test_cli_(
            ["--model", "small", "--language", "en", "--efficient"],
            "corner_cases",
            files=["apollo11.mp3"],
            prefix="stucked_lm",
        )

    def test_punctuation_only(self):

        # When there is only a punctuation detected in a segment, it could cause issue #24
        self._test_cli_(
            ["--model", "medium.en", "--efficient", "--punctuations", "False"],
            "corner_cases",
            files=["empty.wav"],
            prefix="issue24",
        )

    def test_temperature(self):

        self._test_cli_(
            ["--model", "small", "--language", "English",
                "--condition", "False", "--temperature", "0.1", "--efficient"],
            "corner_cases",
            files=["apollo11.mp3"],
            prefix="random.nocond",
        )

        if self.skipLongTests():
            return

        self._test_cli_(
            ["--model", "small", "--language", "en", "--temperature", "0.2", "--efficient"],
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
            ["--model", "medium", "--language", "en", "--condition", "False", "--efficient"],
            "corner_cases",
            files=["music.mp4"],
            prefix="nocond",
        )

        self._test_cli_(
            ["--model", "medium", "--language", "en",
                "--condition", "False", "--temperature", "0.4", "--efficient"],
            "corner_cases",
            files=["music.mp4"],
            prefix="nocond.random",
        )

    def test_large(self):
        if self.skipLongTests():
            return

        self._test_cli_(
            ["--model", "large-v2", "--language", "en",
                "--condition", "False", "--temperature", "0.4", "--efficient"],
            "corner_cases",
            files=["apollo11.mp3"],
            prefix="large",
        )

        if os.path.exists(self.get_data_path("arabic.mp3", check=False)):
            self._test_cli_(
                ["--model", "large-v2", "--language", "Arabic", "--efficient"],
                "corner_cases",
                files=["arabic.mp3"]
            )

    def test_gloria(self):

        for model in ["medium", "large-v2"]:
            for dec in ["efficient", "accurate"]:
                self._test_cli_(
                    ["--model", model, "--language", "en", "--" + dec],
                    "corner_cases",
                    files=["gloria.mp3"],
                    prefix=model + "." + dec,
                )

class TestTranscribeMonolingual(TestHelperCli):

    def test_monolingual_tiny(self):

        files = ["bonjour_vous_allez_bien.mp3"]

        self._test_cli_(
            ["--model", "tiny.en", "--efficient"],
            "tiny.en",
            files=files,
            prefix="efficient",
        )

        self._test_cli_(
            ["--model", "tiny.en", "--accurate"],
            "tiny.en",
            files=files,
            prefix="accurate",
        )

        self._test_cli_(
            ["--model", "tiny.en", "--condition", "False", "--efficient"],
            "tiny.en",
            files=files,
            prefix="nocond",
        )

    def test_monolingual_small(self):

        if os.path.exists(self.get_data_path("arabic.mp3", check=False)):
            self._test_cli_(
                ["--model", "small.en", "--condition", "True", "--efficient"],
                "small.en",
                files=["arabic.mp3"],
                device_specific=True,
            )


class TestTranscribeWithVad(TestHelperCli):

    def test_vad_default(self):
        self._test_cli_(
                ["--model", "tiny", "--accurate", "--language", "en", "--vad", "True", "--verbose", "True"],
                "verbose",
                files=["words.wav"],
                prefix="vad",
                extensions=None,
            )

    def test_vad_custom_silero(self):
        self._test_cli_(
                ["--model", "tiny", "--accurate", "--language", "en", "--vad", "silero:v3.1", "--verbose", "True"],
                "verbose",
                files=["words.wav"],
                prefix="vad_silero3.1",
                extensions=None,
            )
        self._test_cli_(
                ["--model", "tiny", "--accurate", "--language", "en", "--vad", "silero:v3.0", "--verbose", "True"],
                "verbose",
                files=["words.wav"],
                prefix="vad_silero3.0",
                extensions=None,
            )

    def test_vad_custom_auditok(self):
        self._test_cli_(
                ["--model", "tiny", "--language", "en", "--vad", "auditok", "--verbose", "True"],
                "verbose",
                files=["words.wav"],
                prefix="vad_auditok",
                extensions=None,
            )


class TestTranscribeUnspacedLanguage(TestHelperCli):

    def test_japanese(self):

        self._test_cli_(
            ["--model", "tiny", "--efficient"],
            "tiny_auto",
            files=["japanese.mp3"],
            device_specific=True,
        )

        self._test_cli_(
            ["--model", "tiny", "--language", "Japanese", "--efficient"],
            "tiny_auto",
            files=["japanese.mp3"],
            prefix="jp",
            device_specific=True,
        )

        self._test_cli_(
            ["--model", "tiny", "--accurate"],
            "tiny_auto",
            files=["japanese.mp3"],
            prefix="accurate",
            device_specific=True,
        )

        self._test_cli_(
            ["--model", "tiny", "--language", "Japanese", "--accurate"],
            "tiny_auto",
            files=["japanese.mp3"],
            prefix="accurate_jp",
            device_specific=True,
        )

class TestTranscribeFormats(TestHelperCli):

    def test_cli_outputs(self):
        files = ["punctuations.mp3", "bonjour.wav"]
        extensions = ["txt", "srt", "vtt", "words.srt", "words.vtt",
                      "words.json", "csv", "words.csv", "tsv", "words.tsv"]
        opts = ["--model", "medium", "--language", "fr"]

        # An audio / model combination that produces coma
        self._test_cli_(
            opts,
            "punctuations_yes",
            files=files,
            extensions=extensions,
            one_per_call=False,
        )
        self._test_cli_(
            opts + ["--punctuations", "False"],
            "punctuations_no",
            files=files,
            extensions=extensions,
            one_per_call=False,
        )

    def test_verbose(self):

        files = ["bonjour_vous_allez_bien.mp3"]
        opts = ["--model", "tiny", "--verbose", "True"]

        self._test_cli_(
            ["--efficient", *opts],
            "verbose", files=files, extensions=None,
            prefix="efficient.auto",
            device_specific=True,
        )

        self._test_cli_(
            ["--language", "fr", "--efficient", *opts],
            "verbose", files=files, extensions=None,
            prefix="efficient.fr",
            device_specific=True,
        )

        self._test_cli_(
            opts,
            "verbose", files=files, extensions=None,
            prefix="accurate.auto",
            device_specific=True,
        )

        self._test_cli_(
            ["--language", "fr", *opts],
            "verbose", files=files, extensions=None,
            prefix="accurate.fr",
            device_specific=True,
        )

class TestMakeSubtitles(TestHelper):

    def test_make_subtitles(self):

        main_script = self.main_script("make_subtitles.py", "whisper_timestamped_make_subtitles")

        inputs = [
            self.get_data_path("smartphone.mp3.words.json"),
            self.get_data_path("no_punctuations.mp3.words.json", check=True),
            self.get_data_path("yes_punctuations.mp3.words.json", check=True),
        ]

        for i, input in enumerate(inputs):
            filename = os.path.basename(input).replace(".words.json", "")
            for len in 6, 20, 50:
                output_dir = self.get_output_path()
                self.assertRun([main_script, 
                    input if i > 0 else self.get_data_path(), output_dir,
                    "--max_length", str(len),
                ])
                for format in "vtt", "srt",:
                    output_file = os.path.join(output_dir, f"{filename}.{format}")
                    self.assertTrue(os.path.isfile(output_file), msg=f"File {output_file} not found")
                    expected_file = f"split_subtitles/{filename.split('_')[-1]}_{len}.{format}"
                    self.assertNonRegression(output_file, expected_file)
                    os.remove(output_file)
                    self.assertRun([main_script, 
                        input, output_file,
                        "--max_length", str(len),
                    ])
                    self.assertTrue(os.path.isfile(output_file), msg=f"File {output_file} not found")
                    self.assertNonRegression(output_file, expected_file)

class TestHuggingFaceModel(TestHelperCli):

    def test_hugging_face_model(self):

        self._test_cli_(
            ["--model", "qanastek/whisper-tiny-french-cased", "--verbose", "True"],
            "verbose", files=["bonjour.wav"], extensions=None,
            prefix="hf",
            device_specific=True,
        )

        import tempfile
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig
        tempfolder = os.path.join(tempfile.gettempdir(), "tmp_whisper-tiny-french-cased")

        for safe_serialization in False, True,:
            for max_shard_size in "100MB", "10GB", :
                shutil.rmtree(tempfolder, ignore_errors=True)
                model = WhisperForConditionalGeneration.from_pretrained("qanastek/whisper-tiny-french-cased")
                processor = WhisperProcessor.from_pretrained("qanastek/whisper-tiny-french-cased")
                try:
                    model.save_pretrained(tempfolder, safe_serialization=safe_serialization, max_shard_size=max_shard_size)
                    processor.save_pretrained(tempfolder)
                    self._test_cli_(
                        ["--model", tempfolder, "--verbose", "True"],
                        "verbose", files=["bonjour.wav"], extensions=None,
                        prefix="hf",
                        device_specific=True,
                    )
                finally:
                    shutil.rmtree(tempfolder)


# "ZZZ" to run this test at last (because it will fill the CUDA with some memory)
class TestZZZPythonImport(TestHelper):

    def test_python_import(self):

        try:
            import whisper_timestamped
        except ModuleNotFoundError:
            sys.path.append(os.path.realpath(
                os.path.dirname(os.path.dirname(__file__))))
            import whisper_timestamped

        # Test version
        version = whisper_timestamped.__version__
        self.assertTrue(isinstance(version, str))

        (stdout, sterr) = self.assertRun([self.main_script(), "-v"])
        self.assertEqual(stdout.strip(), version)

        model = whisper_timestamped.load_model("tiny")

        # Check processing of different files
        for filename in "bonjour.wav", "laugh1.mp3", "laugh2.mp3":
            res = whisper_timestamped.transcribe(
                model, self.get_data_path(filename))
            if self._can_generate_reference():
                self.assertNonRegression(res, f"tiny_auto/{filename}.words.json")

        for filename in "bonjour.wav", "laugh1.mp3", "laugh2.mp3":
            res = whisper_timestamped.transcribe(
                model, self.get_data_path(filename), language="fr")
            if self._can_generate_reference():
                self.assertNonRegression(res, f"tiny_fr/{filename}.words.json")

    def _can_generate_reference(self):
        return not GENERATE_DEVICE_DEPENDENT or self.get_device_str() != "cpu"

    def test_split_tokens(self):

        import whisper
        whisperversion = whisper.__version__

        import whisper_timestamped as whisper
        from whisper_timestamped.transcribe import split_tokens_on_spaces

        tokenizer = whisper.tokenizer.get_tokenizer(True, language=None)

        # 220 means space
        tokens = [50364, 220, 6455, 11, 2232, 11, 286, 2041, 11, 2232, 11, 8660,
                  291, 808, 493, 220, 365, 11, 220, 445, 718, 505, 458, 13, 220, 50714]

        self.assertEqual(
            split_tokens_on_spaces(tokens, tokenizer),
            (['<|0.00|>', 'So,', 'uh,', 'I', 'guess,', 'uh,', 'wherever', 'you', 'come', 'up', 'with,', 'just', 'let', 'us', 'know.', '<|7.00|>'],
                [['<|0.00|>'],
                [' ', 'So', ','],
                [' uh', ','],
                [' I'],
                [' guess', ','],
                [' uh', ','],
                [' wherever'],
                [' you'],
                [' come'],
                [' up'],
                [' ', ' with', ','],
                [' ', ' just'],
                [' let'],
                [' us'],
                [' know', '.', ' '],
                ['<|7.00|>']],
             [[50364],
                [220, 6455, 11],
                [2232, 11],
                [286],
                [2041, 11],
                [2232, 11],
                [8660],
                [291],
                [808],
                [493],
                [220, 365, 11],
                [220, 445],
                [718],
                [505],
                [458, 13, 220],
                [50714]
              ])
        )

        tokens = [50366, 314, 6, 11771, 17134, 11, 4666, 11, 1022, 220, 875, 2557, 68, 11, 6992, 631, 269, 6, 377, 220, 409, 7282, 1956, 871, 566, 2707, 394, 1956, 256, 622, 8208, 631, 8208, 871, 517, 7282, 1956, 5977, 7418, 371, 1004, 306, 580, 11, 5977, 12, 9498, 9505, 84, 6, 50416]
        self.assertEqual(
            split_tokens_on_spaces(tokens, tokenizer),
                (
                ['<|0.04|>', "T'façon,", 'nous,', 'sur', 'la', 'touche,', 'parce', 'que', "c'est", 'un', 'sport', 'qui', 'est', 'important', 'qui', 'tue', 'deux', 'que', 'deux', 'est', 'un', 'sport', 'qui', 'peut', 'être', 'violent,', 'peut-être', "qu'", '<|1.04|>'],
                [['<|0.04|>'],
                 [' T', "'", 'fa', 'çon', ','],
                    [' nous', ','],
                    [' sur'],
                    [' ', 'la'],
                    [' touch', 'e', ','],
                    [' parce'],
                    [' que'],
                    [' c', "'", 'est'],
                    [' ', 'un'],
                    [' sport'],
                    [' qui'],
                    [' est'],
                    [' im', 'port', 'ant'],
                    [' qui'],
                    [' t', 'ue'],
                    [' deux'],
                    [' que'],
                    [' deux'],
                    [' est'],
                    [' un'],
                    [' sport'],
                    [' qui'],
                    [' peut'],
                    [' être'],
                    [' v', 'io', 'le', 'nt', ','],
                    [' peut', '-', 'être'],
                    [' q', 'u', "'"],
                    ['<|1.04|>']],
                [[50366],
                 [314, 6, 11771, 17134, 11],
                    [4666, 11],
                    [1022],
                    [220, 875],
                    [2557, 68, 11],
                    [6992],
                    [631],
                    [269, 6, 377],
                    [220, 409],
                    [7282],
                    [1956],
                    [871],
                    [566, 2707, 394],
                    [1956],
                    [256, 622],
                    [8208],
                    [631],
                    [8208],
                    [871],
                    [517],
                    [7282],
                    [1956],
                    [5977],
                    [7418],
                    [371, 1004, 306, 580, 11],
                    [5977, 12, 9498],
                    [9505, 84, 6],
                    [50416]]
            )
        )

        tokens = [50364, 220, 220, 6455, 11, 220, 220, 2232, 220, 220, 11, 220, 50714]
        self.assertEqual(
            split_tokens_on_spaces(tokens, tokenizer),
            (['<|0.00|>', 'So,', 'uh', ',', '<|7.00|>'],
            [['<|0.00|>'],
            [' ', ' ', 'So', ','],
            [' ', ' ', ' uh'],
            [' ', ' ', ',', ' '],
            ['<|7.00|>']],
            [[50364], [220, 220, 6455, 11], [220, 220, 2232], [220, 220, 11, 220], [50714]]
            )
        )

        # Careful with the double spaces at the end...
        tokens = [50364, 220, 220, 6455, 11, 220, 220, 2232, 220, 220, 11, 220, 220, 50714]
        self.assertEqual(
            split_tokens_on_spaces(tokens, tokenizer),
            (['<|0.00|>', 'So,', 'uh', ',', '', '<|7.00|>'],
            [['<|0.00|>'],
            [' ', ' ', 'So', ','],
            [' ', ' ', ' uh'],
            [' ', ' ', ','],
            [' ', ' '],
            ['<|7.00|>']],
            [[50364], [220, 220, 6455, 11], [220, 220, 2232], [220, 220, 11], [220, 220], [50714]]
            )
        )

        # Tokens that could be removed
        tokens = [50364, 6024, 95, 8848, 7649, 8717, 38251, 11703, 3224, 51864]
        self.assertEqual(
            split_tokens_on_spaces(tokens, tokenizer),
            (['<|0.00|>', 'الآذان', 'نسمّه', '<|30.00|>'],
                [['<|0.00|>'], ['', ' الآ', 'ذ', 'ان'], [' ن', 'سم', 'ّ', 'ه'], ['<|30.00|>']],
                [[50364], [6024, 95, 8848, 7649], [8717, 38251, 11703, 3224], [51864]]
            )
        )

        # issue #61
        # Special tokens that are not timestamps
        tokens = [50414, 805, 12, 17, 50299, 11, 568, 12, 18, 12, 21, 11, 502, 12, 17, 12, 51464]
        # 50299 is "<|te|>" and appears as ""
        te = ""
        self.assertEqual(
            split_tokens_on_spaces(tokens, tokenizer),
            (['<|1.00|>', f'3-2{te},', '2-3-6,', '1-2-', '<|22.00|>'],
             [['<|1.00|>'], [' 3', '-', '2', f'{te}', ','], [' 2', '-', '3', '-','6', ','], [' 1', '-', '2', '-'], ['<|22.00|>']],
             [[50414], [805, 12, 17, 50299, 11], [568, 12, 18, 12, 21, 11], [502, 12, 17, 12], [51464]])
        )

        tokenizer = whisper.tokenizer.get_tokenizer(False, language="en")

        # Just a punctuation character
        tokens = [50363, 764, 51813]

        _dot = "." if whisperversion < "20230314" else " ."
        self.assertEqual(
            split_tokens_on_spaces(tokens, tokenizer),
            (['<|0.00|>', ".", '<|29.00|>'],
                [['<|0.00|>'], [_dot], ['<|29.00|>']],
                [[50363], [764], [51813]]
            )
        )
