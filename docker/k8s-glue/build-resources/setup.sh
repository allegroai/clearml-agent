#!/bin/bash

chmod +x /root/entrypoint.sh

apt-get update -qqy
apt-get dist-upgrade -qqy
apt-get install -qqy curl unzip less locales

locale-gen en_US.UTF-8

apt-get update -qqy
apt-get install -qqy curl gcc python3-dev python3-pip apt-transport-https lsb-release openssh-client git gnupg
rm -rf /var/lib/apt/lists/*
apt clean

python3 -m pip install -U pip
python3 -m pip install --no-cache-dir clearml-agent
python3 -m pip install -U --no-cache-dir "cryptography>=2.9"
