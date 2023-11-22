#!/bin/bash -x

export CLEARML_FILES_HOST=${CLEARML_FILES_HOST:-$TRAINS_FILES_HOST}

if [ -z "$CLEARML_FILES_HOST" ]; then
    CLEARML_HOST_IP=${CLEARML_HOST_IP:-${TRAINS_HOST_IP:-$(curl -s https://ifconfig.me/ip)}}
fi

export CLEARML_FILES_HOST=${CLEARML_FILES_HOST:-${TRAINS_FILES_HOST:-"http://$CLEARML_HOST_IP:8081"}}
export CLEARML_WEB_HOST=${CLEARML_WEB_HOST:-${TRAINS_WEB_HOST:-"http://$CLEARML_HOST_IP:8080"}}
export CLEARML_API_HOST=${CLEARML_API_HOST:-${TRAINS_API_HOST:-"http://$CLEARML_HOST_IP:8008"}}

echo $CLEARML_FILES_HOST $CLEARML_WEB_HOST $CLEARML_API_HOST 1>&2

if [ -z "$CLEARML_AGENT_NO_UPDATE" ]; then
  if [ -n "$CLEARML_AGENT_UPDATE_REPO" ]; then
    python3 -m pip install -q -U "$CLEARML_AGENT_UPDATE_REPO"
  else
    python3 -m pip install -q -U "clearml-agent${CLEARML_AGENT_UPDATE_VERSION:-$TRAINS_AGENT_UPDATE_VERSION}"
  fi
fi

QUEUE=${K8S_GLUE_QUEUE:-k8s_glue}
MAX_PODS=${K8S_GLUE_MAX_PODS:-2}
EXTRA_ARGS=${K8S_GLUE_EXTRA_ARGS:-}

# shellcheck disable=SC2129
echo "api.credentials.access_key: ${CLEARML_API_ACCESS_KEY}" >> ~/clearml.conf
echo "api.credentials.secret_key: ${CLEARML_API_SECRET_KEY}" >> ~/clearml.conf
echo "api.api_server: ${CLEARML_API_HOST}" >> ~/clearml.conf
echo "api.web_server: ${CLEARML_WEB_HOST}" >> ~/clearml.conf
echo "api.files_server: ${CLEARML_FILES_HOST}" >> ~/clearml.conf

./provider_entrypoint.sh

if [[ -z "${K8S_GLUE_MAX_PODS}" ]]
then
  python3 k8s_glue_example.py --queue ${QUEUE} ${EXTRA_ARGS}
else
  python3 k8s_glue_example.py --queue ${QUEUE} --max-pods ${MAX_PODS} ${EXTRA_ARGS}
fi
