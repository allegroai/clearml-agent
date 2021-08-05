#!/bin/bash

chmod +x /root/entrypoint.sh

apt-get update -y
apt-get dist-upgrade -y
apt-get install -y curl unzip less locales

locale-gen en_US.UTF-8

apt-get install -y curl python3-pip git
python3 -m pip install -U pip
python3 -m pip install clearml-agent
python3 -m pip install -U "cryptography>=2.9"
