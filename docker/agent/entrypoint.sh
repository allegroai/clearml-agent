#!/bin/sh

LOWER_PIP_UPDATE_VERSION="$(echo "$PIP_UPDATE_VERSION" | tr '[:upper:]' '[:lower:]')"
LOWER_TRAINS_AGENT_UPDATE_VERSION="$(echo "$TRAINS_AGENT_UPDATE_VERSION" | tr '[:upper:]' '[:lower:]')"

if [ "$LOWER_PIP_UPDATE_VERSION" = "yes" ] || [ "$LOWER_PIP_UPDATE_VERSION" = "true" ] ; then
    python3 -m pip install -U pip
elif [ ! -z "$LOWER_PIP_UPDATE_VERSION" ] ; then
    python3 -m pip install pip$LOWER_PIP_UPDATE_VERSION ;
fi

echo "TRAINS_AGENT_UPDATE_VERSION = $LOWER_TRAINS_AGENT_UPDATE_VERSION"
if [ "$LOWER_TRAINS_AGENT_UPDATE_VERSION" = "yes" ] || [ "$LOWER_TRAINS_AGENT_UPDATE_VERSION" = "true" ] ; then
    python3 -m pip install trains-agent -U
elif [ ! -z "$LOWER_TRAINS_AGENT_UPDATE_VERSION" ] ; then
    python3 -m pip install trains-agent$LOWER_TRAINS_AGENT_UPDATE_VERSION ;
fi

python3 -m trains_agent daemon --docker "$TRAINS_AGENT_DEFAULT_BASE_DOCKER" --force-current-version $TRAINS_AGENT_EXTRA_ARGS