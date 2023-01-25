import os

from setuptools import setup, find_packages

install_requires = open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines()

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

setup(
    name="whisper-timestamped",
    py_modules=["whisper_timestamped"],
    version=version,
    description="Add to OpenAI Whisper the capability to give word timestamps",
    readme="README.md",
    python_requires=">=3.8",
    author="Jeronymous",
    url="https://github.com/Jeronymous/whisper-timestamped",
    license=license,
    packages=find_packages(exclude=["tests*"]),
    install_requires=install_requires,
    entry_points = {
        'console_scripts': ['whisper_timestamped=whisper_timestamped.transcribe:cli'],
    },
    include_package_data=True,
    extras_require={'dev': ['matplotlib']},
)
