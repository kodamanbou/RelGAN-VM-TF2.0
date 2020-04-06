FROM tensorflow/tensorflow:2.1.0-gpu-py3
MAINTAINER Hideo Kodama <kodamanbou0424@gmail.com>

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y zip \
                       unzip \
                       ffmpeg \
                       nano

RUN pip3 install librosa \
                 pyworld \
                 matplotlib \
                 tqdm \
                 tensorflow-addons==0.8.3

RUN mkdir work
WORKDIR work

ENV TF_FORCE_GPU_ALLOW_GROWTH=true
EXPOSE 6006:6006