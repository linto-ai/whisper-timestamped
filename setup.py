import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="whisper-timestamped",
    py_modules=["whisper_timestamped"],
    version="1.0",
    description="Add to OpenAI Whisper the capability to give word timestamps",
    readme="README.md",
    python_requires=">=3.7",
    author="Jernoymous",
    url="https://github.com/Jeronymous/whisper-timestamped",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    entry_points = {
        'console_scripts': ['whisper_timestamped=whisper_timestamped.transcribe:cli'],
    },
    include_package_data=True,
    extras_require={'dev': ['matplotlib']}, # TODO: 'pytest'
)
