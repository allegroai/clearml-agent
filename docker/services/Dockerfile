# syntax = docker/dockerfile
FROM ubuntu:18.04

WORKDIR /usr/agent

COPY . /usr/agent

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV PYTHONIOENCODING=UTF-8

RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install -y locales

RUN locale-gen en_US.UTF-8

RUN apt-get install -y curl python3-pip git
RUN curl -sSL https://get.docker.com/ | sh
RUN python3 -m pip install -U pip
RUN python3 -m pip install clearml-agent
RUN python3 -m pip install -U "cryptography>=2.9"

ENTRYPOINT ["/usr/agent/entrypoint.sh"]
