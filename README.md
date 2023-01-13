# whisper-timestamped
Get word timestamps from OpenAI Whisper transcription.

## Description
Whisper is a multi-lingual robust speech recognition model, trained by OpenAI,
that achieves state-of-the-art in many languages.

The Whisper models were trained to predict approximative timestamps on speech segments (most of the times with 1 sec accuracy),
but cannot originally predict word timestamps.

This repository proposes an approach to recover word timestamps, and give more accurate estimation of speech segments,
when transcribing with Whipser models.

The approach is based on approach dynamic time warping applied to attention weights,
as done by [this notebook by Jong Wook Kim](https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/notebooks/Multilingual_ASR.ipynb).
The main addition to this notebook is that no additional inference steps are required to predict word timestamps.
Word alignment is done on the fly, after each speech segment is decoded.
In particular, little additional memory is used with respect to the regular use of the model.

## Acknowlegment.
* [whisper](https://github.com/openai/whisper) Whisper speech recognition (License MIT).
* [dtw-python](https://pypi.org/project/dtw-python) Comprehensive implementation of Dynamic Time Warping algorithms.
