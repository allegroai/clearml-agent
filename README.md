# Allegro Trains Agent
## Deep Learning DevOps For Everyone - Now supporting all platforms (Linux, macOS, and Windows)

"All the Deep-Learning DevOps your research needs, and then some... Because ain't nobody got time for that"

[![GitHub license](https://img.shields.io/github/license/allegroai/trains-agent.svg)](https://img.shields.io/github/license/allegroai/trains-agent.svg)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/trains-agent.svg)](https://img.shields.io/pypi/pyversions/trains-agent.svg)
[![PyPI version shields.io](https://img.shields.io/pypi/v/trains-agent.svg)](https://img.shields.io/pypi/v/trains-agent.svg)
[![PyPI status](https://img.shields.io/pypi/status/trains-agent.svg)](https://pypi.python.org/pypi/trains-agent/)

### Help improve Trains by filling our 2-min [user survey](https://allegro.ai/lp/trains-user-survey/)

**Trains Agent is an AI experiment cluster solution.**

It is a zero configuration fire-and-forget execution agent, which combined with trains-server provides a full AI cluster solution.

**Full AutoML in 5 steps** 
1. Install the [Trains Server](https://github.com/allegroai/trains-agent) (or use our [open server](https://demoapp.trains.allegro.ai))
2. `pip install trains-agent` ([install](#installing-the-trains-agent) the Trains Agent on any GPU machine: on-premises / cloud / ...)
3. Add [Trains](https://github.com/allegroai/trains) to your code with just 2 lines & run it once (on your machine / laptop)
4. Change the [parameters](#using-the-trains-agent) in the UI & schedule for [execution](#using-the-trains-agent) (or automate with an [AutoML pipeline](#automl-and-orchestration-pipelines-))
5. :chart_with_downwards_trend: :chart_with_upwards_trend: :eyes:  :beer:


**Using the Trains Agent, you can now set up a dynamic cluster with \*epsilon DevOps**

*epsilon - Because we are scientists :triangular_ruler: and nothing is really zero work

(Experience Trains live at [https://demoapp.trains.allegro.ai](https://demoapp.trains.allegro.ai))
<a href="https://demoapp.trains.allegro.ai"><img src="https://raw.githubusercontent.com/allegroai/trains-agent/9f1e86c1ca45c984ee13edc9353c7b10c55d7257/docs/screenshots.gif" width="100%"></a>

## Simple, Flexible Experiment Orchestration
**The Trains Agent was built to address the DL/ML R&D DevOps needs:**

* Easily add & remove machines from the cluster
* Reuse machines without the need for any dedicated containers or images
* **Combine GPU resources across any cloud and on-prem**
* **No need for yaml/json/template configuration of any kind**
* **User friendly UI**
* Manageable resource allocation that can be used by researchers and engineers
* Flexible and controllable scheduler with priority support
* Automatic instance spinning in the cloud **(coming soon)**


## But ... K8S?
We think Kubernetes is awesome.  
Combined with KubeFlow it is a robust solution for production-grade DevOps.  
We've observed, however, that it can be a bit of an overkill as an R&D DL/ML solution.
If you are considering K8S for your research, also consider that you will soon be managing **hundreds** of containers...

In our experience, handling and building the environments, having to package every experiment in a docker, managing those hundreds (or more) containers and building pipelines on top of it all, is very complicated (also, it’s usually out of scope for the research team, and overwhelming even for the DevOps team).

We feel there has to be a better way, that can be just as powerful for R&D and at the same time allow integration with K8S **when the need arises**.  
(If you already have a K8S cluster for AI, detailed instructions on how to integrate Trains into your K8S cluster are [here](https://github.com/allegroai/trains-server-k8s/tree/master/trains-server-chart) with included [helm chart](https://github.com/allegroai/trains-server-helm))


## Using the Trains Agent
**Full scale HPC with a click of a button**

The Trains Agent is a job scheduler that listens on job queue(s), pulls jobs, sets the job environments, executes the job and monitors its progress.

Any 'Draft' experiment can be scheduled for execution by a Trains agent.

A previously run experiment can be put into 'Draft' state by either of two methods:
* Using the **'Reset'** action from the experiment right-click context menu in the
  Trains UI - This will clear any results and artifacts the previous run had created.
* Using the **'Clone'** action from the experiment right-click context menu in the
  Trains UI - This will create a new 'Draft' experiment with the same configuration as the original experiment.

An experiment is scheduled for execution using the **'Enqueue'** action from the experiment
 right-click context menu in the Trains UI and selecting the execution queue.

See [creating an experiment and enqueuing it for execution](#from-scratch).

Once an experiment is enqueued, it will be picked up and executed by a Trains agent monitoring this queue.

The Trains UI Workers & Queues page provides ongoing execution information:
  - Workers Tab: Monitor you cluster
    - Review available resources
    - Monitor machines statistics (CPU / GPU / Disk / Network)
  - Queues Tab:
    - Control the scheduling order of jobs
    - Cancel or abort job execution
    - Move jobs between execution queues

### What The Trains Agent Actually Does
The Trains Agent executes experiments using the following process:
  - Create a new virtual environment (or launch the selected docker image)
  - Clone the code into the virtual-environment (or inside the docker)
  - Install python packages based on the package requirements listed for the experiment
    - Special note for PyTorch: The Trains Agent will automatically select the
      torch packages based on the CUDA_VERSION environment variable of the machine
  - Execute the code, while monitoring the process
  - Log all stdout/stderr in the Trains UI, including the cloning and installation process, for easy debugging
  - Monitor the execution and allow you to manually abort the job using the Trains UI (or, in the unfortunate case of a code crash, catch the error and signal the experiment has failed)

### System Design & Flow
```text
                                                                              +-----------------+
                                                                              |  GPU  Machine   |
Development Machine                                                           |                 |
+------------------------+                                                    | +-------------+ |
|    Data Scientist's    |                            +--------------+        | |Trains Agent | |
|      DL/ML Code        |                            |    WEB UI    |        | |             | |
|                        |                            |              |        | | +---------+ | |
|                        |                            |              |        | | |  DL/ML  | | |
|                        |                            +--------------+        | | |  Code   | | |
|                        |       User Clones Exp #1  / . . . . . . . /        | | |         | | |
| +-------------------+  |           into Exp #2    / . . . . . . . /         | | +---------+ | |
| |      Trains       |  |         +---------------/-_____________-/          | |             | |
| +---------+---------+  |         |                                          | |      ^      | |
+-----------|------------+         |                                          | +------|------+ |
            |                      |                                          +--------|--------+
 Auto-Magically                    |                                                   |
 Creates Exp #1                    |                                      The Trains Agent
             \          User Change Hyper-Parameters                      Pulls Exp #2, setup the
             |                     |                                      environment & clone code.
             |                     |                                      Start execution with the
+------------|------------+        |            +--------------------+    new set of Hyper-Parameters.
|  +---------v---------+  |        |            |   Trains Server    |                 |
|  | Experiment #1     |  |        |            |                    |                 |
|  +-------------------+  |        |            |  Execution Queue   |                 |
|            ||           |        |            |                    |                 |
|  +-------------------+<----------+            |                    |                 |
|  |                   |  |                     |                    |                 |
|  | Experiment #2     |  |                     |                    |                 |
|  +-------------------<------------\           |                    |                 |
|                         |          ------------->---------------+  |                 |
|                         |  User Send Exp #2   | |Execute Exp #2 +--------------------+
|                         |  For Execution      | +---------------+  |
|     Trains Server       |                     |                    |
+-------------------------+                     +--------------------+
```

### Installing the Trains Agent

```bash
pip install trains-agent
```

### Trains Agent Usage Examples

Full Interface and capabilities are available with
```bash
trains-agent --help
trains-agent daemon --help
```

### Configuring the Trains Agent

```bash
trains-agent init
```

Note: The Trains Agent uses a cache folder to cache pip packages, apt packages and cloned repositories. The default Trains Agent cache folder is `~/.trains`

See full details in your configuration file at `~/trains.conf`

Note: The **Trains agent** extends the **Trains** configuration file `~/trains.conf`
They are designed to share the same configuration file, see example [here](docs/trains.conf)

### Running the Trains Agent

For debug and experimentation, start the Trains agent in `foreground` mode, where all the output is printed to screen
```bash
trains-agent daemon --queue default --foreground
```

For actual service mode, all the stdout will be stored automatically into a temporary file (no need to pipe)
Notice: with `--detached` flag, the *trains-agent* will be running in the background
```bash
trains-agent daemon --detached --queue default
```

GPU allocation is controlled via the standard OS environment `NVIDIA_VISIBLE_DEVICES` or `--gpus` flag (or disabled with `--cpu-only`).

If no flag is set, and `NVIDIA_VISIBLE_DEVICES` variable doesn't exist, all GPU's will be allocated for the `trains-agent` <br>
If `--cpu-only` flag is set, or `NVIDIA_VISIBLE_DEVICES` is an empty string (""), no gpu will be allocated for the `trains-agent`

Example: spin two agents, one per gpu on the same machine:
Notice: with `--detached` flag, the *trains-agent* will be running in the background
```bash
trains-agent daemon --detached --gpus 0 --queue default
trains-agent daemon --detached --gpus 1 --queue default
```

Example: spin two agents, pulling from dedicated `dual_gpu` queue, two gpu's per agent
```bash
trains-agent daemon --detached --gpus 0,1 --queue dual_gpu
trains-agent daemon --detached --gpus 2,3 --queue dual_gpu
```

#### Starting the Trains Agent in docker mode

For debug and experimentation, start the Trains agent in `foreground` mode, where all the output is printed to screen
```bash
trains-agent daemon --queue default --docker --foreground
```

For actual service mode, all the stdout will be stored automatically into a file (no need to pipe)
Notice: with `--detached` flag, the *trains-agent* will be running in the background
```bash
trains-agent daemon --detached --queue default --docker
```

Example: spin two agents, one per gpu on the same machine, with default nvidia/cuda docker:
```bash
trains-agent daemon --detached --gpus 0 --queue default --docker nvidia/cuda
trains-agent daemon --detached --gpus 1 --queue default --docker nvidia/cuda
```

Example: spin two agents, pulling from dedicated `dual_gpu` queue, two gpu's per agent, with default nvidia/cuda docker:
```bash
trains-agent daemon --detached --gpus 0,1 --queue dual_gpu --docker nvidia/cuda
trains-agent daemon --detached --gpus 2,3 --queue dual_gpu --docker nvidia/cuda
```

#### Starting the Trains Agent - Priority Queues

Priority Queues are also supported, example use case:

High priority queue: `important_jobs`  Low priority queue: `default`
```bash
trains-agent daemon --queue important_jobs default
```
The **Trains Agent** will first try to pull jobs from the `important_jobs` queue, only then it will fetch a job from the `default` queue.

Adding queues, managing job order within a queue and moving jobs between queues, is available using the Web UI, see example on our [open server](https://demoapp.trains.allegro.ai/workers-and-queues/queues)

#### Stopping the Trains Agent

To stop a **Trains Agent** running in the background, run the same command line used to start the agent with `--stop` appended.  
For example, to stop the first of the above shown same machine, single gpu agents:
```bash
trains-agent daemon --detached --gpus 0 --queue default --docker nvidia/cuda --stop
```

## How do I create an experiment on the Trains Server? <a name="from-scratch"></a>
* Integrate [Trains](https://github.com/allegroai/trains) with your code
* Execute the code on your machine (Manually / PyCharm / Jupyter Notebook)
* As your code is running, **Trains** creates an experiment logging all the necessary execution information:
  - Git repository link and commit ID (or an entire jupyter notebook)
  - Git diff (we’re not saying you never commit and push, but still...)
  - Python packages used by your code (including specific versions used)
  - Hyper-Parameters
  - Input Artifacts

  You now have a 'template' of your experiment with everything required for automated execution

* In the Trains UI, Right click on the experiment and select 'clone'. A copy of your experiment will be created.
* You now have a new draft experiment cloned from your original experiment, feel free to edit it
  - Change the Hyper-Parameters
  - Switch to the latest code base of the repository
  - Update package versions
  - Select a specific docker image to run in (see docker execution mode section)
  - Or simply change nothing to run the same experiment again...
* Schedule the newly created experiment for execution: Right-click the experiment and select 'enqueue'

## Trains-Agent Services Mode <a name="services"></a> 

Trains-Agent Services is a special mode of Trains-Agent that provides the ability to launch long-lasting jobs 
that previously had to be executed on local / dedicated machines. It allows a single agent to 
launch multiple dockers (Tasks) for different use cases. To name a few use cases, auto-scaler service (spinning instances 
when the need arises and the budget allows), Controllers (Implementing pipelines and more sophisticated DevOps logic),
Optimizer (such as Hyper-parameter Optimization or sweeping), and Application (such as interactive Bokeh apps for 
increased data transparency)

Trains-Agent Services mode will spin **any** task enqueued into the specified queue. 
Every task launched by Trains-Agent Services will be registered as a new node in the system, 
providing tracking and transparency capabilities. 
Currently trains-agent in services-mode supports cpu only configuration. Trains-agent services mode can be launched alongside GPU agents.

```bash
trains-agent daemon --services-mode --detached --queue services --create-queue --docker ubuntu:18.04 --cpu-only
```

**Note**: It is the user's responsibility to make sure the proper tasks are pushed into the specified queue. 


## AutoML and Orchestration Pipelines <a name="automl-pipes"></a>
The Trains Agent can also be used to implement AutoML orchestration and Experiment Pipelines in conjunction with the Trains package.

Sample AutoML & Orchestration examples can be found in the Trains [example/automation](https://github.com/allegroai/trains/tree/master/examples/automation) folder.

AutoML examples
  - [Toy Keras training experiment](https://github.com/allegroai/trains/blob/master/examples/optimization/hyper-parameter-optimization/base_template_keras_simple.py)
    - In order to create an experiment-template in the system, this code must be executed once manually
  - [Random Search over the above Keras experiment-template](https://github.com/allegroai/trains/blob/master/examples/automation/manual_random_param_search_example.py)
    - This example will create multiple copies of the Keras experiment-template, with different hyper-parameter combinations

Experiment Pipeline examples
  - [First step experiment](https://github.com/allegroai/trains/blob/master/examples/automation/task_piping_example.py)
    - This example will "process data", and once done, will launch a copy of the 'second step' experiment-template
  - [Second step experiment](https://github.com/allegroai/trains/blob/master/examples/automation/toy_base_task.py)
    - In order to create an experiment-template in the system, this code must be executed once manually

## License

Apache License, Version 2.0 (see the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0.html) for more information)
