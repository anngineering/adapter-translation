# Refer to README.md for commonly used PyTorch images
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV DEBIAN_FRONTEND noninteractive

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -y install make \
    && apt-get install python3-dev -y \
    && apt-get install gcc -y \
    && apt-get install git -y \
    && apt-get -y install build-essential git apt-utils gcc libpq-dev libsndfile1 ffmpeg sox wget python3-pip vim \
    && rm -rf /var/lib/apt/lists/*

RUN apt install -y git

RUN pip install -U pip

ADD build/requirements.txt .
RUN pip uninstall -y -r requirements.txt --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install git+https://github.com/huggingface/transformers
RUN python -m spacy download en_core_web_lg

ADD /src /src

WORKDIR /src