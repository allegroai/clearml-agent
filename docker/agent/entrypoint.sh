#!/bin/sh

LOWER_PIP_UPDATE_VERSION="$(echo "$PIP_UPDATE_VERSION" | tr '[:upper:]' '[:lower:]')"
LOWER_CLEARML_AGENT_UPDATE_VERSION="$(echo "${CLEARML_AGENT_UPDATE_VERSION:-$TRAINS_AGENT_UPDATE_VERSION}" | tr '[:upper:]' '[:lower:]')"

if [ "$LOWER_PIP_UPDATE_VERSION" = "yes" ] || [ "$LOWER_PIP_UPDATE_VERSION" = "true" ] ; then
    python3 -m pip install -U pip
elif [ ! -z "$LOWER_PIP_UPDATE_VERSION" ] ; then
    python3 -m pip install pip$LOWER_PIP_UPDATE_VERSION ;
fi

echo "CLEARML_AGENT_UPDATE_VERSION = $LOWER_CLEARML_AGENT_UPDATE_VERSION"
if [ "$LOWER_CLEARML_AGENT_UPDATE_VERSION" = "yes" ] || [ "$LOWER_CLEARML_AGENT_UPDATE_VERSION" = "true" ] ; then
    python3 -m pip install clearml-agent -U
elif [ ! -z "$LOWER_CLEARML_AGENT_UPDATE_VERSION" ] ; then
    python3 -m pip install clearml-agent$LOWER_CLEARML_AGENT_UPDATE_VERSION ;
fi

python3 -m clearml_agent daemon --docker "${CLEARML_AGENT_DEFAULT_BASE_DOCKER:-$TRAINS_AGENT_DEFAULT_BASE_DOCKER}" --force-current-version ${CLEARML_AGENT_EXTRA_ARGS:-$TRAINS_AGENT_EXTRA_ARGS}