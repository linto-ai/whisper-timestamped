# whisper-timestamped

Multilingual Automatic Speech Recognition with word-level timestamps and confidence.

* [Description](#description)
   * [Notes on other approaches](#notes-on-other-approaches)
* [Installation](#installation)
   * [First installation](#first-installation)
   * [Light installation for CPU](#light-installation-for-cpu)
   * [Upgrade to the latest version](#upgrade-to-the-latest-version)
* [Usage](#usage)
   * [Python](#python)
   * [Command line](#command-line)
   * [Plotting word alignment](#plotting-word-alignment)
   * [Example output](#example-output)
   * [Options that may improve results](#options-that-may-improve-results)
* [Acknowlegment](#acknowlegment)
* [Citations](#citations)

## Description
[Whisper](https://openai.com/blog/whisper/) is a set of multi-lingual robust speech recognition models, trained by OpenAI,
that achieve state-of-the-art in many languages.
Whisper models were trained to predict approximative timestamps on speech segments (most of the times with 1 sec accuracy),
but cannot originally predict word timestamps.
This repository proposes an implementation to **predict word timestamps, and give more accurate estimation of speech segments, when transcribing with Whipser models**.
Besides, a confidence score is assigned to each word and each segment (both computed as "exp(mean(log probas))" on the probabilities of subword tokens).

The approach is based on approach Dynamic Time Warping (DTW) applied to cross-attention weights,
as done by [this notebook by Jong Wook Kim](https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/notebooks/Multilingual_ASR.ipynb).
There are some additions to this notebook:
* The start/end estimation is more accurate.
* Confidence scores are assigned to each word.
* **If possible (without beam search...)**, there no additional inference steps are required to predict word timestamps (word alignment is done on the fly, after each speech segment is decoded).
* There is a special care about memory usage: `whisper-timestamped` is able to process long files, with little additional memory with respect to the regular use of Whisper model.

`whisper-timestamped` is an extension of [`openai-whisper`](https://pypi.org/project/whisper-openai/) python package
and is meant to compatible with any version of `openai-whisper`.

### Notes on other approaches

An alternative relevant approach to recover word-level timestamps consists in using wav2vec models that predict characters,
as successfully implemented in [whisperX](https://github.com/m-bain/whisperX).
But these approaches have several drawbacks, which does not have approaches based on cross-attention weights such as `whisper_timestamped`.
These drawbacks are:
* The need to find one wav2vec model per language to support, which badly scales to the multi-lingual capabilities of Whisper.
* The need to handle (at least) one additional neural network (wav2vec model), which consumes memory.
* The need to normalize characters in whisper transcription to match the character set of wav2vec model.
This involves awkward language-dependent conversions, like converting numbers to words ("2" -> "two"), symbols to words ("%" -> "percent", "€" -> "euro(s)")...
* The lack of robustness around speech disfluencies (fillers, hesitations, repeated words...) that are usually removed by Whisper.

An alternative approach, that does not require an additional model, is to look at the probabilities of timestamp tokens
estimated by the Whisper model after each (sub)word token is predicted.
It was implemented for instance in [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and [stable-ts](https://github.com/jianfch/stable-ts).
But this approach lacks of robustness, because Whisper models do not have been trained to output meaningful timestamps after each word.
Whisper models tend to predict timestamps only after a certain number of words have been predicted (typically at the end of a sentence),
and the probability distribution of timestamps outside this condition may be inaccurate.
In practice, these methods can produce results that are totally out-of-sync on some periods of time (we observed that especially when there is jingle music).
Also the timestamp precision of Whisper models tend to be rounded to 1 second (as in many video subtitles), which is too inaccurate for words, and reaching a better accuracy is tricky.

## Installation

### First installation

Requirements:
* `python3` (version higher or equal to 3.7, at least 3.9 is recommended)
* `ffmpeg` (see instructions for installation on the [whisper repository](https://github.com/openai/whisper)

You can install `whisper-timestamped` either by using pip:
```bash
pip3 install git+https://github.com/linto-ai/whisper-timestamped
```
or by cloning this repository and running installation:
```bash
git clone https://github.com/linto-ai/whisper-timestamped
cd whisper-timestamped/
python3 setup.py install
```

If you want to plot alignement between audio timestamps and words (as in [this section](#plotting-word-alignment)), you also need matplotlib
```bash
pip3 install matplotlib
```

A docker image of about 9GB can be built using:
```bash
git clone https://github.com/linto-ai/whisper-timestamped
cd whisper-timestamped/
docker build -t whisper_timestamped:latest .
```

### Light installation for CPU

If you don't have GPU (or don't want to use it), then you don't need to install CUDA dependencies.
You should then just install a light version of torch **before** installing whisper-timestamped, for instance as follows:
```bash
pip3 install \
     torch==1.13.1+cpu \
     torchaudio==0.13.1+cpu \
     -f https://download.pytorch.org/whl/torch_stable.html
```

A specific docker image of about 3.5GB can also be built using:
```bash
git clone https://github.com/linto-ai/whisper-timestamped
cd whisper-timestamped/
docker build -t whisper_timestamped_cpu:latest -f Dockerfile.cpu .
```

### Upgrade to the latest version

When using pip, the library can be updated to the latest version using
```
pip3 install --upgrade --no-deps --force-reinstall git+https://github.com/linto-ai/whisper-timestamped
```

A specific version of `openai-whisper` can be used by running, for example:
```bash
pip3 install openai-whisper==20230124
```

## Usage

### Python

In python, you can use the function `whisper_timestamped.transcribe()` that is similar to the fonction `whisper.transcribe()`
```python
import whisper_timestamped
help(whisper_timestamped.transcribe)
```
The main difference with `whisper.transcribe()` is that
the output will include a key `"words"` for all segments, with the word start and end position. Note that word will include punctuation. See example [below](#example-output).

Besides, default decoding options are different, in order to favour efficient decoding (greedy decoding instead of beam search, and no temperature sampling fallback).
To have same default as in `whisper`, use ```beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)```.

There are also additional options related to word alignement.

In general, by importing `whisper_timestamped` instead of `whisper` in your python script, it should do the job, if you use `transcribe(model, ...)` instead of `model.transcribe(...)`:
```
import whisper_timestamped as whisper

audio = whisper.load_audio("AUDIO.wav")

model = whisper.load_model("tiny", device="cpu")

result = whisper.transcribe(model, audio, language="fr")

import json
print(json.dumps(result, indent = 2, ensure_ascii = False))
```

### Command line

You can also use `whisper_timestamped` on the command line, similarly to `whisper`. See help with:
```bash
whisper_timestamped --help
```

The main differences with `whisper` CLI are:
* Output files:
  * The output JSON contains word timestamps and confidence scores. See example [below](#example-output).
  * There is an additional CSV output format
  * For SRT, VTT, TSV formats, there will be additional files saved with word timestamps
* Some default options are different:
  * By default, no output folder is set: Use `--output_dir .` for Whisper default
  * By default, there is no verbose: Use `--verbose True` for Whisper default
  * By default, beam search decoding and temperature sampling fallback are disabled, to favour an efficient decoding.
    To set the same as Whisper default, you can use `--accurate` (which is an alias for ```--beam_size 5 --temperature_increment_on_fallback 0.2 --best_of 5```).
* There are some additional specific options:
  <!-- * `--efficient` to use a faster greedy decoding (without beam search neither several sampling at each step),
  which enables a special path where word timestamps are computed on the fly (no need to run inference twice).
  Note that transcription results might be significantly worse on challenging audios with this option. -->
  * `--compute_confidence` to enable/disable the computation of confidence scores for each word.
  * `--punctuations_with_words` to decide whether punctuation marks should be included or not with preceding words.

An example command line to process several files with the `tiny` model and output results in the current folder as whisper would do by default:
```
whisper_timestamped audio1.flac audio2.mp3 audio3.wav --model tiny --output_dir .
```

### Plot of word alignment

Note that you can use option `plot_word_alignment` of python function `whisper_timestamped.transcribe()`, or option `--plot` of `whisper_timestamped` CLI in order to see the word alignment for each segment.

![Example alignement](figs/example_alignement_plot.png)

* The upper plot represents the transformation of cross-attention weights that is used for the alignement with Dynamic Time Warping.
The abscissa represents the time and the ordinate represents the predicted tokens; with special timestamp tokens at first and at last, and then (sub)words and punctuations in the middle.
* The lower plot is a MFCC representation of the input signal (features used by Whisper, based on Mel-frequency cepstrum).
* The vertical dotted red lines show where the word boundaries are found (with punctuation marks "glued" with the previous word).

### Example output

Here is an example output of `whisper_timestamped.transcribe()`, that can be seen by using CLI
```bash
whisper_timestamped AUDIO_FILE.wav --model tiny --language fr
```
```json
{
  "text": " Bonjour! Est-ce que vous allez bien?",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.5,
      "end": 1.2,
      "text": " Bonjour!",
      "tokens": [ 25431, 2298 ],
      "temperature": 0.0,
      "avg_logprob": -0.6674491882324218,
      "compression_ratio": 0.8181818181818182,
      "no_speech_prob": 0.10241222381591797,
      "confidence": 0.51,
      "words": [
        {
          "text": "Bonjour!",
          "start": 0.5,
          "end": 1.2,
          "confidence": 0.51
        }
      ]
    },
    {
      "id": 1,
      "seek": 200,
      "start": 2.02,
      "end": 4.48,
      "text": " Est-ce que vous allez bien?",
      "tokens": [ 50364, 4410, 12, 384, 631, 2630, 18146, 3610, 2506, 50464 ],
      "temperature": 0.0,
      "avg_logprob": -0.43492694334550336,
      "compression_ratio": 0.7714285714285715,
      "no_speech_prob": 0.06502953916788101,
      "confidence": 0.595,
      "words": [
        {
          "text": "Est-ce",
          "start": 2.02,
          "end": 3.78,
          "confidence": 0.441
        },
        {
          "text": "que",
          "start": 3.78,
          "end": 3.84,
          "confidence": 0.948
        },
        {
          "text": "vous",
          "start": 3.84,
          "end": 4.0,
          "confidence": 0.935
        },
        {
          "text": "allez",
          "start": 4.0,
          "end": 4.14,
          "confidence": 0.347
        },
        {
          "text": "bien?",
          "start": 4.14,
          "end": 4.48,
          "confidence": 0.998
        }
      ]
    }
  ],
  "language": "fr"
}
```

### Options that may improve results

Here are some options not abled by default that might improve results.

#### Accurate Whisper transcription

As mentioned before, some decoding options are disabled by default for offering a better efficiency.
But the quality of the transcription can be impacted.
To run with the options that have the best chance to provide a good transcription, use the following options.
* In python:
```python
results = whisper_timestamped.transcribe(model, audio, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), ...)
```
* In the command line:
```bash
whisper_timestamped --accurate ...
```

#### Running Voice Activity Detection (VAD) before sending to Whisper

Whisper models can "hallucinate" text when a segment without speech is given.
This can be avoided by running VAD and gluing speech segments together before transcribing with the Whisper model.
This is possible in `whisper-timestamped`.
* In python:
```python
results = whisper_timestamped.transcribe(model, audio, vad=True, ...)
```
* In the command line:
```bash
whisper_timestamped --vad True ...
```

#### Detecting disfluencies

Whisper models tend to remove speech disfluencies (filler words, hesitations, repetitions, ...).
Without precautions, the disfluencies that are not transcribed will have an influence on the timestamp of the word that follows: the timestamp of the beginning of the word will actually be the timestamp of the beginning of the disfluencies.
`whisper-timestamped` can implement some heuristics to avoid that.
* In python:
```python
results = whisper_timestamped.transcribe(model, audio, detect_disfluencies=True, ...)
```
* In the command line:
```bash
whisper_timestamped --detect_disfluencies True ...
```
**Important:** Note that when using this options, possible disfluencies will appear in the transcription as a special "`[*]`" word.


## Acknowlegment
* [whisper](https://github.com/openai/whisper): Whisper speech recognition (License MIT).
* [dtw-python](https://pypi.org/project/dtw-python): Dynamic Time Warping (License GPL v3).

## Citations
If you use this in your research, just cite the repo,

```bibtex
@misc{lintoai2023whispertimestamped,
  title={whisper-timestamped},
  author={Louradour, J{\'e}r{\^o}me},
  journal={GitHub repository},
  year={2023},
  publisher={GitHub},
  howpublished = {\url{https://github.com/linto-ai/whisper-timestamped}}
}
```

as well as OpenAI Whisper paper,

```bibtex
@article{radford2022robust,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```

and this paper for Dynamic-Time-Warping

```bibtex
@article{JSSv031i07,
  title={Computing and Visualizing Dynamic Time Warping Alignments in R: The dtw Package},
  author={Giorgino, Toni},
  journal={Journal of Statistical Software},
  year={2009},
  volume={31},
  number={7},
  doi={10.18637/jss.v031.i07}
}
```
