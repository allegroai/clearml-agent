FROM ubuntu:18.04

USER root
WORKDIR /root

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV PYTHONIOENCODING=UTF-8

COPY ../build-resources/setup.sh /root/setup.sh
RUN /root/setup.sh

COPY ./setup_aws.sh /root/setup_aws.sh
RUN /root/setup_aws.sh

COPY ../build-resources/entrypoint.sh /root/entrypoint.sh
COPY ./provider_entrypoint.sh /root/provider_entrypoint.sh
COPY ./build-resources/k8s_glue_example.py /root/k8s_glue_example.py
COPY ./clearml.conf /root/clearml.conf

ENTRYPOINT ["/root/entrypoint.sh"]