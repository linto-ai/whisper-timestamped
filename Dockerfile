FROM python:3.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg

RUN python3 -m pip install --upgrade pip

# Python installation
WORKDIR /usr/src/app
COPY whisper_timestamped /usr/src/app/whisper_timestamped
COPY requirements.txt /usr/src/app/requirements.txt
COPY setup.py /usr/src/app/setup.py

RUN cd /usr/src/app/ && python3 setup.py install
RUN rm -R /usr/src/app/requirements.txt /usr/src/app/setup.py /usr/src/app/whisper_timestamped

# WTF?
RUN ln -s \
   /usr/local/lib/python3.9/site-packages/nvidia_cudnn_*/nvidia/cudnn \
   /usr/local/lib/python3.9/site-packages/nvidia_cublas_*/nvidia/

COPY tests /usr/src/app/tests

ENTRYPOINT ["/bin/bash"]