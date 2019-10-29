# TRAINS Agent
## Deep Learning DevOps For Everyone

"Because you can setup a cluster with only two lines!"

[![GitHub license](https://img.shields.io/github/license/allegroai/trains-agent.svg)](https://img.shields.io/github/license/allegroai/trains-agent.svg)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/trains-agent.svg)](https://img.shields.io/pypi/pyversions/trains-agent.svg)
[![PyPI version shields.io](https://img.shields.io/pypi/v/trains-agent.svg)](https://img.shields.io/pypi/v/trains-agent.svg)
[![PyPI status](https://img.shields.io/pypi/status/trains-agent.svg)](https://pypi.python.org/pypi/trains-agent/)

TRAINS Agent is an AI experiment cluster solution.

It is a zero configuration fire-and-forget execution agent and combined with trains-server it is a full AI cluster solution.

**Using the TRAINS Agent, you can now setup a dynamic cluster with only two lines!**

(Experience TRAINS live at [https://demoapp.trains.allegro.ai](https://demoapp.trains.allegro.ai))
<a href="https://demoapp.trains.allegro.ai"><img src="https://github.com/allegroai/trains-agent/blob/master/docs/screenshots.gif?raw=true" width="100%"></a>

## Simple, Flexible Experiment Orchestration
**The TRAINS Agent was built to address the DL/ML R&D DevOps needs:**

* Easily add & remove machines from the cluster
* Reuse machines without the need for any dedicated containers or images
* **Combine on-prem GPU resources with any cloud GPU resources**
* **No need for yaml/json/template configuration of any kind**
* **User friendly UI**
* Manageable resource allocation that can be used by researchers and engineers
* Flexible and controllable scheduler with priority support
* Automatic instance spinning in the cloud **(coming soon)**

### Integrating with Kubernetes 
K8S is awesome. It is a great tool and combined with KubeFlow it's a robust solution for production. Let us stress that point again - *"For Production"*.
It was never designed to help or facilitate R&D efforts of DL/ML. Having to package every experiment in a docker, managing those hundreds (or more) containers and building pipelines on top of it all is complicated (it’s usually out of scope for the research team, and overwhelming even for the DevOps team).

We feel there has to be a better way, that can be just as powerful for R&D and at the same time allow integration with K8S **when the need arises**. If you already have a K8S cluster for AI, detailed instructions on how to integrate TRAINS into your K8S cluster is *coming soon*.


## Using the TRAINS Agent
**Full scale HPC with a click of a button**

TRAINS Agent is a job scheduler that listens on job queue(s), pulls jobs, sets the job environments, executes the job and monitors its progress.

Any 'Draft' experiment can be scheduled for execution by a TRAINS agent.

A previously run experiment can be put into 'Draft' state by either of two methods:
* Using the **'Reset'** action from the experiment right-click context menu in the
  TRAINS UI - This will clear any results and artifacts the previous run had created.
* Using the **'Clone'** action from the experiment right-click context menu in the
  TRAINS UI - This will create a new 'Draft' experiment with the same configuration as the original experiment.

An experiment is scheduled for execution using the **'Enqueue'** action from the experiment
 right-click context menu in the TRAINS UI and selecting the execution queue.
 
See [creating an experiment, and enqueuing it for execution](#from-scratch).

Once an experiment is enqueued, it will be picked up and executed by a TRAINS agent monitoring this queue.

The TRAINS UI Workers & Queues page provides ongoing execution information:
  - Workers Tab: Monitor you cluster
    - Review available resources
    - Monitor machines statistics (CPU / GPU / Disk / Network)
  - Queues Tab: 
    - Control the scheduling order of jobs
    - Cancel or abort job execution
    - Move jobs between execution queues

### What The TRAINS Agent Actually Does
The TRAINS agent executes experiments using the following process:
  - Create a new virtual environment (or launch the selected docker image)
  - Clone the code into the virtual-environment (or inside the docker)
  - Install python packages based on the package requirements listed for the experiment
    - Special note for PyTorch, The TRAINS agent will automatically select the 
      torch packages based on the CUDA_VERSION environment of the machine
  - Execute the code, while monitoring the process
  - Log all stdout/stderr in the TRAINS UI, including the cloning and installation process, for easy debugging
  - Monitor the execution and allow you to manually abort the job using the TRAINS UI (or, in the unfortunate case of a code crash, catch the error and signal the experiment has failed)
  
### System Design & Flow
```text
                                                                                                     +-----------------+              
                                                                                                     |  GPU  Machine   |              
    Development Machine                                                                              |                 |              
    +------------------------+                                                                       |                 |              
    |    Data Scientist's    |                            +--------------+                           | +-------------+ |              
    |      DL/ML Code        |                            |    WEB UI    |                           | |TRAINS Agent | |              
    |                        |                            |              |                           | |             | |              
    |                        |                            |              |                           | | +---------+ | |              
    |                        |                            +--------------+                           | | |  DL/ML  | | |              
    |                        |       User Clones Exp #1  / . . . . . . . /                           | | |  Code   | | |              
    | +-------------------+  |           into Exp #2    / . . . . . . . /                            | | |         | | |              
    | |      TRAINS       |  |         +---------------/-_____________-/                             | | |         | | |              
    | +---------+---------+  |         |                                                             | | +----^----+ | |              
    +-----------|------------+         |                                                             | +------|------+ |              
                |                      |                                                             +--------|--------+              
 Auto-Magically |                      |                                                                      |                       
 Creates Exp #1 |                      |                                                                      |                       
                 \          User Change Hyper-Parameters                                                      |                       
                 |                     |                                                                      |                       
                 |                     |                                                                      |                       
    +------------|------------+        |            +--------------------+                                    |                       
    |  +---------v---------+  |        |            |   TRAINS-SERVER    |                                    |                       
    |  | Experiment #1     |  |        |            |                    |                                    |                       
    |  +-------------------+  |        |            |  Execution Queue   |                                    |                       
    |            ||           |        |            |                    |                                    |                       
    |  +-------------------+<----------+            |                    |    The TRAINS Agent                |                       
    |  |                   |  |                     |                    |    Pulls Exp #2                    |                       
    |  | Experiment #2     |  |                     |                    |    Sets the environment and code   |                       
    |  +-------------------<------------\           |                    |    Start execution with the        |                       
    |                         |          ------------->---------------+  |    new set of Hyper-Parameters     |                       
    |                         |  User Send Exp #2   | |Execute Exp #2 +---------------------------------------+                       
    |                         |  For Execution      | +---------------+  |                                                            
    |     TRAINS-SERVER       |                     |                    |                                                            
    +-------------------------+                     +--------------------+                                                            
```

### Installing the TRAINS Agent

```bash
pip install trains_agent
```

### TRAINS Agent Usage Examples

Full Interface and capabilities are available with
```bash
trains-agent --help
trains-agent daemon --help
```

### Configuring the TRAINS Agent

```bash
trains-agent init
```

Note: The TRAINS agent uses a cache folder to cache pip packages, apt packages and cloned repositories. The default TRAINS Agent cache folder is `~/.trains`

See full details in your configuration file at `~/trains.conf`

Note: The **TRAINS agent** extends the **TRAINS** configuration file `~/trains.conf`
They are designed to share the same configuration file, see example [here](docs/trains.conf)

### Running the TRAINS Agent

For debug and experimentation, start the TRAINS agent in `foreground` mode, where all the output is printed to screen
```bash
trains-agent daemon --queue default --foreground
```

For actual service mode, all the stdout will be stored automatically into a temporary file (no need to pipe)
```bash
trains-agent daemon --queue default
```

#### Starting the TRAINS Agent in docker mode

For debug and experimentation, start the TRAINS agent in `foreground` mode, where all the output is printed to screen
```bash
trains-agent daemon --queue default --docker --foreground
```

For actual service mode, all the stdout will be stored automatically into a file (no need to pipe)
```bash
trains-agent daemon --queue default --docker
```

#### Starting the TRAINS Agent - Priority Queues

Priority Queues are also supported, example use case: 

High priority queue: `important_jobs`  Low priority queue: `default`
```bash
trains-agent daemon --queue important_jobs default
```
The **TRAINS agent** will first try to pull jobs from the `important_jobs` queue, only then it will fetch a job from the `default` queue.

# AutoML and Orchestration Pipelines <a name="automl-pipes"></a>
The TRAINS Agent can also implement AutoML orchestration and Experiment Pipelines in conjunction with the TRAINS package.

Sample AutoML & Orchestration examples can be found in the TRAINS [example/automl](https://github.com/allegroai/trains/tree/master/examples/automl) folder.

AutoML examples
  - [Toy Keras training experiment](https://github.com/allegroai/trains/blob/master/examples/automl/automl_base_template_keras_simple.py)
    - In order to create an experiment-template in the system, this code must be executed once manually 
  - [Random Search over the above Keras experiment-template](https://github.com/allegroai/trains/blob/master/examples/automl/automl_random_search_example.py)
    - This example will create multiple copies of the Keras experiment-template, with different hyper-parameter combinations

Experiment Pipeline examples
  - [First step experiment](https://github.com/allegroai/trains/blob/master/examples/automl/task_piping_example.py)
    - This example will "process data", and once done, will launch a copy of the 'second step' experiment-template
  - [Second step experiment](https://github.com/allegroai/trains/blob/master/examples/automl/toy_base_task.py)
    - In order to create an experiment-template in the system, this code must be executed once manually
     
 # How do I create an experiment on the TRAINS server? <a name="from-scratch"></a>
* Integrate [TRAINS](https://github.com/allegroai/trains) with your code
* Execute the code on your machine (Manually / PyCharm / Jupyter Notebook)
* As your code is running, **TRAINS** creates an experiment logging all the necessary execution information:
  - Git repository link and commit ID (or an entire jupyter notebook)
  - Git diff (we’re not saying you never commit and push, but still...)
  - Python packages used by your code (including specific versions used)
  - Hyper-Parameters
  - Input Artifacts
  
  You now have a 'template' of your experiment with everything required for automated execution
  
* In the TRAINS UI, Right click on the experiment and select 'clone'. A copy of your experiment will be created.
* You now have a new draft experiment cloned from your original experiment, feel free to edit it
  - Change the Hyper-Parameters
  - Switch to the latest code base of the repository
  - Update package versions
  - Select a specific docker image to run in (see docker execution mode section)
  - Or simply change nothing to run the same experiment again...
* Send the newly created experiment for execution, right-click the experiment and select 'enqueue'
