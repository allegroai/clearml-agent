<div align="center">

<img src="https://github.com/allegroai/clearml-agent/blob/master/docs/clearml_agent_logo.png?raw=true" width="250px">

**ClearML Agent - ML-Ops made easy  
ML-Ops scheduler & orchestration solution supporting Linux, macOS and Windows**

[![GitHub license](https://img.shields.io/github/license/allegroai/clearml-agent.svg)](https://img.shields.io/github/license/allegroai/clearml-agent.svg)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/clearml-agent.svg)](https://img.shields.io/pypi/pyversions/clearml-agent.svg)
[![PyPI version shields.io](https://img.shields.io/pypi/v/clearml-agent.svg)](https://img.shields.io/pypi/v/clearml-agent.svg)
[![PyPI Downloads](https://pepy.tech/badge/clearml-agent/month)](https://pypi.org/project/clearml-agent/)
[![Artifact Hub](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/allegroai)](https://artifacthub.io/packages/search?repo=allegroai)
</div>

---

### ClearML-Agent

#### *Formerly known as Trains Agent*

* Run jobs (experiments) on any local or cloud based resource
* Implement optimized resource utilization policies
* Deploy execution environments with either virtualenv or fully docker containerized with zero effort
* Launch-and-Forget service containers
* [Cloud autoscaling](https://clear.ml/docs/latest/docs/guides/services/aws_autoscaler)
* [Customizable cleanup](https://clear.ml/docs/latest/docs/guides/services/cleanup_service)
* Advanced [pipeline building and execution](https://clear.ml/docs/latest/docs/guides/frameworks/pytorch/notebooks/table/tabular_training_pipeline)

It is a zero configuration fire-and-forget execution agent, providing a full ML/DL cluster solution.

**Full Automation in 5 steps**

1. ClearML Server [self-hosted](https://github.com/allegroai/clearml-server)
   or [free tier hosting](https://app.clear.ml)
2. `pip install clearml-agent` ([install](#installing-the-clearml-agent) the ClearML Agent on any GPU machine:
   on-premises / cloud / ...)
3. Create a [job](https://clear.ml/docs/latest/docs/apps/clearml_task) or
   add [ClearML](https://github.com/allegroai/clearml) to your code with just 2 lines of code
4. Change the [parameters](#using-the-clearml-agent) in the UI & schedule for [execution](#using-the-clearml-agent) (or
   automate with an [AutoML pipeline](#automl-and-orchestration-pipelines-))
5. :chart_with_downwards_trend: :chart_with_upwards_trend: :eyes:  :beer:

"All the Deep/Machine-Learning DevOps your research needs, and then some... Because ain't nobody got time for that"

**Try ClearML now** [Self Hosted](https://github.com/allegroai/clearml-server)
or [Free tier Hosting](https://app.clear.ml)
<a href="https://app.clear.ml"><img src="https://github.com/allegroai/clearml-agent/blob/master/docs/screenshots.gif?raw=true" width="100%"></a>

### Simple, Flexible Experiment Orchestration

**The ClearML Agent was built to address the DL/ML R&D DevOps needs:**

* Easily add & remove machines from the cluster
* Reuse machines without the need for any dedicated containers or images
* **Combine GPU resources across any cloud and on-prem**
* **No need for yaml / json / template configuration of any kind**
* **User friendly UI**
* Manageable resource allocation that can be used by researchers and engineers
* Flexible and controllable scheduler with priority support
* Automatic instance spinning in the cloud

**Using the ClearML Agent, you can now set up a dynamic cluster with \*epsilon DevOps**

*epsilon - Because we are :triangular_ruler: and nothing is really zero work

### Kubernetes Integration (Optional)

We think Kubernetes is awesome, but it should be a choice. We designed `clearml-agent` so you can run bare-metal or
inside a pod with any mix that fits your environment.

Find Dockerfiles in the [docker](./docker) dir and a helm Chart in https://github.com/allegroai/clearml-helm-charts

#### Benefits of integrating existing K8s with ClearML-Agent

- ClearML-Agent adds the missing scheduling capabilities to K8s
- Allowing for more flexible automation from code
- A programmatic interface for easier learning curve (and debugging)
- Seamless integration with ML/DL experiment manager
- Web UI for customization, scheduling & prioritization of jobs

**Two K8s integration flavours**

- Spin ClearML-Agent as a long-lasting service pod:
    - Use [clearml-agent](https://hub.docker.com/r/allegroai/clearml-agent) docker image
    - map docker socket into the pod (soon replaced by [podman](https://github.com/containers/podman))
    - Allow the clearml-agent to manage sibling dockers
    - Benefits: full use of the ClearML scheduling, no need to worry about wrong container images / lost pods etc.
    - Downside: sibling containers
- Kubernetes Glue, map ClearML jobs directly to K8s jobs:
    - Run the [clearml-k8s glue](https://github.com/allegroai/clearml-agent/blob/master/examples/k8s_glue_example.py) on
      a K8s cpu node
    - The clearml-k8s glue pulls jobs from the ClearML job execution queue and prepares a K8s job (based on provided
      yaml template)
    - Inside the pod itself the clearml-agent will install the job (experiment) environment and spin and monitor the
      experiment's process
    - Benefits: Kubernetes full view of all running jobs in the system
    - Downside: No real scheduling (k8s scheduler), no docker image verification (post-mortem only)

### Using the ClearML Agent

**Full scale HPC with a click of a button**

The ClearML Agent is a job scheduler that listens on job queue(s), pulls jobs, sets the job environments, executes the
job and monitors its progress.

Any 'Draft' experiment can be scheduled for execution by a ClearML agent.

A previously run experiment can be put into 'Draft' state by either of two methods:

* Using the **'Reset'** action from the experiment right-click context menu in the ClearML UI - This will clear any
  results and artifacts the previous run had created.
* Using the **'Clone'** action from the experiment right-click context menu in the ClearML UI - This will create a new 
  'Draft' experiment with the same configuration as the original experiment.

An experiment is scheduled for execution using the **'Enqueue'** action from the experiment right-click context menu in
the ClearML UI and selecting the execution queue.

See [creating an experiment and enqueuing it for execution](#from-scratch).

Once an experiment is enqueued, it will be picked up and executed by a ClearML Agent monitoring this queue.

The ClearML UI Workers & Queues page provides ongoing execution information:

- Workers Tab: Monitor you cluster
    - Review available resources
    - Monitor machines statistics (CPU / GPU / Disk / Network)
- Queues Tab:
    - Control the scheduling order of jobs
    - Cancel or abort job execution
    - Move jobs between execution queues

#### What The ClearML Agent Actually Does

The ClearML Agent executes experiments using the following process:

- Create a new virtual environment (or launch the selected docker image)
- Clone the code into the virtual-environment (or inside the docker)
- Install python packages based on the package requirements listed for the experiment
    - Special note for PyTorch: The ClearML Agent will automatically select the torch packages based on the CUDA_VERSION
      environment variable of the machine
- Execute the code, while monitoring the process
- Log all stdout/stderr in the ClearML UI, including the cloning and installation process, for easy debugging
- Monitor the execution and allow you to manually abort the job using the ClearML UI (or, in the unfortunate case of a
  code crash, catch the error and signal the experiment has failed)

#### System Design & Flow

<img src="https://github.com/allegroai/clearml-agent/blob/master/docs/clearml_architecture.png" width="100%" alt="clearml-architecture">

#### Installing the ClearML Agent

```bash
pip install clearml-agent
```

#### ClearML Agent Usage Examples

Full Interface and capabilities are available with

```bash
clearml-agent --help
clearml-agent daemon --help
```

#### Configuring the ClearML Agent

```bash
clearml-agent init
```

Note: The ClearML Agent uses a cache folder to cache pip packages, apt packages and cloned repositories. The default
ClearML Agent cache folder is `~/.clearml`.

See full details in your configuration file at `~/clearml.conf`.

Note: The **ClearML Agent** extends the **ClearML** configuration file `~/clearml.conf`.
They are designed to share the same configuration file, see example [here](docs/clearml.conf)

#### Running the ClearML Agent

For debug and experimentation, start the ClearML agent in `foreground` mode, where all the output is printed to screen:

```bash
clearml-agent daemon --queue default --foreground
```

For actual service mode, all the stdout will be stored automatically into a temporary file (no need to pipe).
Notice: with `--detached` flag, the *clearml-agent* will be running in the background

```bash
clearml-agent daemon --detached --queue default
```

GPU allocation is controlled via the standard OS environment `NVIDIA_VISIBLE_DEVICES` or `--gpus` flag (or disabled
with `--cpu-only`).

If no flag is set, and `NVIDIA_VISIBLE_DEVICES` variable doesn't exist, all GPUs will be allocated for
the `clearml-agent`. <br>
If `--cpu-only` flag is set, or `NVIDIA_VISIBLE_DEVICES="none"`, no gpu will be allocated for
the `clearml-agent`.

Example: spin two agents, one per GPU on the same machine:

Notice: with `--detached` flag, the *clearml-agent* will run in the background

```bash
clearml-agent daemon --detached --gpus 0 --queue default
clearml-agent daemon --detached --gpus 1 --queue default
```

Example: spin two agents, pulling from dedicated `dual_gpu` queue, two GPUs per agent

```bash
clearml-agent daemon --detached --gpus 0,1 --queue dual_gpu
clearml-agent daemon --detached --gpus 2,3 --queue dual_gpu
```

##### Starting the ClearML Agent in docker mode

For debug and experimentation, start the ClearML agent in `foreground` mode, where all the output is printed to screen

```bash
clearml-agent daemon --queue default --docker --foreground
```

For actual service mode, all the stdout will be stored automatically into a file (no need to pipe).
Notice: with `--detached` flag, the *clearml-agent* will run in the background

```bash
clearml-agent daemon --detached --queue default --docker
```

Example: spin two agents, one per gpu on the same machine, with default `nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04`
docker:

```bash
clearml-agent daemon --detached --gpus 0 --queue default --docker nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04
clearml-agent daemon --detached --gpus 1 --queue default --docker nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04
```

Example: spin two agents, pulling from dedicated `dual_gpu` queue, two GPUs per agent, with default 
`nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04` docker:

```bash
clearml-agent daemon --detached --gpus 0,1 --queue dual_gpu --docker nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04
clearml-agent daemon --detached --gpus 2,3 --queue dual_gpu --docker nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04
```

##### Starting the ClearML Agent - Priority Queues

Priority Queues are also supported, example use case:

High priority queue: `important_jobs`, low priority queue: `default`

```bash
clearml-agent daemon --queue important_jobs default
```

The **ClearML Agent** will first try to pull jobs from the `important_jobs` queue, and only if it is empty, the agent 
will try to pull from the `default` queue.

Adding queues, managing job order within a queue, and moving jobs between queues, is available using the Web UI, see
example on our [free server](https://app.clear.ml/workers-and-queues/queues)

##### Stopping the ClearML Agent

To stop a **ClearML Agent** running in the background, run the same command line used to start the agent with `--stop`
appended. For example, to stop the first of the above shown same machine, single gpu agents:

```bash
clearml-agent daemon --detached --gpus 0 --queue default --docker nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04 --stop
```

### How do I create an experiment on the ClearML Server? <a name="from-scratch"></a>

* Integrate [ClearML](https://github.com/allegroai/clearml) with your code
* Execute the code on your machine (Manually / PyCharm / Jupyter Notebook)
* As your code is running, **ClearML** creates an experiment logging all the necessary execution information:
    - Git repository link and commit ID (or an entire jupyter notebook)
    - Git diff (we’re not saying you never commit and push, but still...)
    - Python packages used by your code (including specific versions used)
    - Hyperparameters
    - Input artifacts

  You now have a 'template' of your experiment with everything required for automated execution

* In the ClearML UI, right-click on the experiment and select 'clone'. A copy of your experiment will be created.
* You now have a new draft experiment cloned from your original experiment, feel free to edit it
    - Change the hyperparameters
    - Switch to the latest code base of the repository
    - Update package versions
    - Select a specific docker image to run in (see docker execution mode section)
    - Or simply change nothing to run the same experiment again...
* Schedule the newly created experiment for execution: right-click the experiment and select 'enqueue'

### ClearML-Agent Services Mode <a name="services"></a>

ClearML-Agent Services is a special mode of ClearML-Agent that provides the ability to launch long-lasting jobs that
previously had to be executed on local / dedicated machines. It allows a single agent to launch multiple dockers (Tasks)
for different use cases: 
* Auto-scaler service (spinning instances when the need arises and the budget allows)
* Controllers (Implementing pipelines and more sophisticated DevOps logic)
* Optimizer (such as Hyperparameter Optimization or sweeping)
* Application (such as interactive Bokeh apps for increased data transparency)

ClearML-Agent Services mode will spin **any** task enqueued into the specified queue. Every task launched by
ClearML-Agent Services will be registered as a new node in the system, providing tracking and transparency capabilities.
Currently, clearml-agent in services-mode supports CPU only configuration. ClearML-Agent services mode can be launched
alongside GPU agents.

```bash
clearml-agent daemon --services-mode --detached --queue services --create-queue --docker ubuntu:18.04 --cpu-only
```

**Note**: It is the user's responsibility to make sure the proper tasks are pushed into the specified queue.

### AutoML and Orchestration Pipelines <a name="automl-pipes"></a>

The ClearML Agent can also be used to implement AutoML orchestration and Experiment Pipelines in conjunction with the
ClearML package.

Sample AutoML & Orchestration examples can be found in the
ClearML [example/automation](https://github.com/allegroai/clearml/tree/master/examples/automation) folder.

AutoML examples:

- [Toy Keras training experiment](https://github.com/allegroai/clearml/blob/master/examples/optimization/hyper-parameter-optimization/base_template_keras_simple.py)
    - In order to create an experiment-template in the system, this code must be executed once manually
- [Random Search over the above Keras experiment-template](https://github.com/allegroai/clearml/blob/master/examples/automation/manual_random_param_search_example.py)
    - This example will create multiple copies of the Keras experiment-template, with different hyperparameter
      combinations

Experiment Pipeline examples:

- [First step experiment](https://github.com/allegroai/clearml/blob/master/examples/automation/task_piping_example.py)
    - This example will "process data", and once done, will launch a copy of the 'second step' experiment-template
- [Second step experiment](https://github.com/allegroai/clearml/blob/master/examples/automation/toy_base_task.py)
    - In order to create an experiment-template in the system, this code must be executed once manually

### License

Apache License, Version 2.0 (see the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0.html) for more information)
