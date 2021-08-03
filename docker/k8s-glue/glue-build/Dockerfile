FROM ubuntu:18.04

USER root
WORKDIR /root

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV PYTHONIOENCODING=UTF-8

COPY ./entrypoint.sh /root/entrypoint.sh
COPY ./k8s_glue_example.py /root/k8s_glue_example.py
COPY ./setup.sh /root/setup.sh
COPY ./clearml.conf /root/clearml.conf

RUN /root/setup.sh

ENTRYPOINT ["/root/entrypoint.sh"]