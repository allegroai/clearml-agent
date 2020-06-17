#!/bin/sh

if [ -z "$TRAINS_FILES_HOST" ]; then
    TRAINS_HOST_IP=${TRAINS_HOST_IP:-$(curl -s https://ifconfig.me/ip)}
fi

TRAINS_FILES_HOST=${TRAINS_FILES_HOST:-"http://$TRAINS_HOST_IP:8081"}
TRAINS_WEB_HOST=${TRAINS_WEB_HOST:-"http://$TRAINS_HOST_IP:8080"}
TRAINS_API_HOST=${TRAINS_API_HOST:-"http://$TRAINS_HOST_IP:8008"}

echo $TRAINS_FILES_HOST $TRAINS_WEB_HOST $TRAINS_API_HOST 1>&2

python3 -m pip install -q -U "trains-agent${TRAINS_AGENT_UPDATE_VERSION}"
trains-agent daemon --services-mode --queue services --create-queue --docker $TRAINS_AGENT_DEFAULT_BASE_DOCKER --cpu-only $TRAINS_AGENT_EXTRA_ARGS
