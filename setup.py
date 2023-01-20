import os

from setuptools import setup, find_packages

install_requires = open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines()

setup(
    name="whisper-timestamped",
    py_modules=["whisper_timestamped"],
    version="1.5.4",
    description="Add to OpenAI Whisper the capability to give word timestamps",
    readme="README.md",
    python_requires=">=3.7",
    author="Jeronymous",
    url="https://github.com/Jeronymous/whisper-timestamped",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=install_requires,
    entry_points = {
        'console_scripts': ['whisper_timestamped=whisper_timestamped.transcribe:cli'],
    },
    include_package_data=True,
    extras_require={'dev': ['matplotlib']},
)
