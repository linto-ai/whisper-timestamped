import os

from setuptools import setup, find_packages

install_requires = [
    "Cython",
    "dtw-python",
    "openai-whisper",
]

required_packages_filename = os.path.join(os.path.dirname(__file__), "requirements.txt")
if os.path.exists(required_packages_filename):
    install_requires2 = [l.strip() for l in open(required_packages_filename).readlines()]
    assert install_requires == install_requires2, f"requirements.txt is not up-to-date: {install_requires} != {install_requires2}"

version = None
license = None
with open(os.path.join(os.path.dirname(__file__), "whisper_timestamped", "transcribe.py")) as f:
    for line in f:
        if line.strip().startswith("__version__"):
            version = line.split("=")[1].strip().strip("\"'")
            if version and license:
                break
        if line.strip().startswith("__license__"):
            license = line.split("=")[1].strip().strip("\"'")
            if version and license:
                break
assert version and license

description="Multi-lingual Automatic Speech Recognition (ASR) based on Whisper models, with accurate word timestamps, access to language detection confidence, several options for Voice Activity Detection (VAD), and more."

setup(
    name="whisper-timestamped",
    py_modules=["whisper_timestamped"],
    version=version,
    description=description,
    long_description=description+"\nSee https://github.com/linto-ai/whisper-timestamped for more information.",
    long_description_content_type='text/markdown',
    python_requires=">=3.7",
    author="Jeronymous",
    url="https://github.com/linto-ai/whisper-timestamped",
    license=license,
    packages=find_packages(exclude=["tests*"]),
    install_requires=install_requires,
    entry_points = {
        'console_scripts': [
            'whisper_timestamped=whisper_timestamped.transcribe:cli',
            'whisper_timestamped_make_subtitles=whisper_timestamped.make_subtitles:cli'
        ],
    },
    include_package_data=True,
    extras_require={
        'dev': ['matplotlib==3.7.4', 'transformers'],
        'vad_silero': ['onnxruntime', 'torchaudio'],
        'vad_auditok': ['auditok'],
        'test': ['jsonschema'],
    },
)
