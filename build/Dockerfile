# Refer to README.md for commonly used PyTorch images
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

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
    && apt-get install gcc git wget make -y \
    && rm -rf /var/cache/apt/archives/


RUN mkdir /data
ADD data /data

ADD build /build
WORKDIR /build

RUN make

ENV DISPLAY=:99

ADD /src /src

WORKDIR /src