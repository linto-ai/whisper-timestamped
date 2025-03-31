FROM python:3.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg portaudio19-dev

RUN python3 -m pip install --upgrade pip

# Python installation
WORKDIR /usr/src/app

# Note: First installing the python requirements permits to save time when re-building after a source change.
COPY requirements.txt /usr/src/app/requirements.txt
RUN cd /usr/src/app/ && pip3 install -r requirements.txt

# Copy source
COPY setup.py /usr/src/app/setup.py
COPY whisper_timestamped /usr/src/app/whisper_timestamped

# Install
RUN cd /usr/src/app/ && pip3 install ".[dev]"
RUN cd /usr/src/app/ && pip3 install ".[vad_silero]"
RUN cd /usr/src/app/ && pip3 install ".[vad_auditok]"
RUN cd /usr/src/app/ && pip3 install ".[test]"

# Cleanup
RUN rm -R /usr/src/app/requirements.txt /usr/src/app/setup.py /usr/src/app/whisper_timestamped

# Copy tests
COPY tests /usr/src/app/tests

ENTRYPOINT ["/bin/bash"]