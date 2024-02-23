from clearml_agent.helper.environment import EnvEntry

ENV_START_AGENT_SCRIPT_PATH = EnvEntry("CLEARML_K8S_GLUE_START_AGENT_SCRIPT_PATH", default="~/__start_agent__.sh")
"""
Script path to use when creating the bash script to run the agent inside the scheduled pod's docker container. 
Script will be appended to the specified file.
"""

ENV_DEFAULT_EXECUTION_AGENT_ARGS = EnvEntry("K8S_GLUE_DEF_EXEC_AGENT_ARGS", default="--full-monitoring --require-queue")
ENV_POD_AGENT_INSTALL_ARGS = EnvEntry("K8S_GLUE_POD_AGENT_INSTALL_ARGS", default="", lstrip=False)
ENV_POD_MONITOR_LOG_BATCH_SIZE = EnvEntry("K8S_GLUE_POD_MONITOR_LOG_BATCH_SIZE", default=5, converter=int)
