from clearml_agent.definitions import EnvironmentConfig

ENV_START_AGENT_SCRIPT_PATH = EnvironmentConfig('CLEARML_K8S_GLUE_START_AGENT_SCRIPT_PATH')
"""
Script path to use when creating the bash script to run the agent inside the scheduled pod's docker container. 
Script will be appended to the specified file.
"""
