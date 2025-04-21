import os
from setuptools import setup, find_packages
import re

def get_requirements(filename="requirements.txt"):
    """Reads requirements from a requirements file."""
    requirements = []
    full_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

def get_metadata(field):
    """Reads metadata fields like __version__, __license__ from __init__.py"""
    init_path = os.path.join(os.path.dirname(__file__), "whisper_timestamped", "__init__.py")
    metadata_value = None
    if os.path.exists(init_path):
        with open(init_path, 'r', encoding='utf-8') as f:
            init_contents = f.read()
            match = re.search(r"^__" + re.escape(field) + r"__\s*=\s*['\"]([^'\"]*)['\"]", init_contents, re.M)
            if match:
                metadata_value = match.group(1)
            else:
                print(f"Warning: Could not find __{field}__ in {init_path}")
    else:
        print(f"Warning: {init_path} not found.")
    return metadata_value

install_requires = get_requirements()

version = get_metadata("version")
license = get_metadata("license")

if not version:
    raise RuntimeError("Version information not found in whisper_timestamped/__init__.py")

if not license:
    print("Warning: License information not found in whisper_timestamped/__init__.py.")

long_description = ""
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    print("Warning: README.md not found.")


description = "Multilingual automatic speech recognition (ASR) with new speaker segmentation (NSS) and word-level timestamps (WLT)"

setup(
    name="urro_whisper",
    author="urroxyz",
    url="https://github.com/urroxyz/whisper",
    # --------------------
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.7",
    license=license,
    packages=find_packages(exclude=["tests*"]),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
