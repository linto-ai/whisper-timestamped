FROM python:3.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg

RUN python3 -m pip install --upgrade pip

# Python installation
WORKDIR /usr/src/app

# Note: First installing the python requirements permits to save time when re-building after a source change.
#       Also, when not installing the python requirements before the "setup install", there are these issues:
#
# 1. This seems to be needed (WTF?):
# # RUN ln -s \
# #   /usr/local/lib/python3.9/site-packages/nvidia_cudnn_*/nvidia/cudnn \
# #   /usr/local/lib/python3.9/site-packages/nvidia_cublas_*/nvidia/
#
# 2. Sometimes this runtime error occurs:
# #   File "/usr/local/lib/python3.9/site-packages/openai_whisper-20230124-py3.9.egg/whisper/transcribe
# # .py", line 84, in transcribe                                                                       
# #     mel = log_mel_spectrogram(audio)                                                               
# #   File "/usr/local/lib/python3.9/site-packages/openai_whisper-20230124-py3.9.egg/whisper/audio.py",
# #  line 116, in log_mel_spectrogram                                                                  
# #     magnitudes = stft[..., :-1].abs() ** 2                                                         
# # RuntimeError: Error in dlopen for library libnvrtc.so.11.2and libnvrtc-d833c4f3.so.11.2            
COPY requirements.txt /usr/src/app/requirements.txt
RUN cd /usr/src/app/ && pip3 install -r requirements.txt

# Copy source
COPY setup.py /usr/src/app/setup.py
COPY whisper_timestamped /usr/src/app/whisper_timestamped

# Install
RUN cd /usr/src/app/ && python3 setup.py install

# Cleanup
RUN rm -R /usr/src/app/requirements.txt /usr/src/app/setup.py /usr/src/app/whisper_timestamped

# Copy tests
COPY tests /usr/src/app/tests

ENTRYPOINT ["/bin/bash"]