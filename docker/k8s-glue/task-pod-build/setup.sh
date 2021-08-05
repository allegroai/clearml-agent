echo 'Binary::apt::APT::Keep-Downloaded-Packages \"true\";' > /etc/apt/apt.conf.d/docker-clean
chown -R root /root/.cache/pip

apt-get update -y
apt-get dist-upgrade -y
apt-get install -y git libsm6 libxext6 libxrender-dev libglib2.0-0 curl python3-pip

python3 -m pip install -U pip
python3 -m pip install clearml-agent
python3 -m pip install -U "cryptography>=2.9"
