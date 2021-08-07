## Roojoom project

This repository contains the code for the Y-DATA project with Roojoom. The goal of the project was to build a model that takes into account contextual information + historical data, in order to predict the action with the highest chance of solving the problem. The models takes into account the sequence of actions as well.

### Installation

In order to run and work with this repository, you'd need to install the following tools:
* [Docker](https://www.docker.com)
* [DVC](https://dvc.org)
* [Git](https://git-scm.com/)

Moreover, for our development environment, we used [ml-workspace](https://github.com/ml-tooling/ml-workspace). <b>You do not need to install it at this point.</b>


### Usage
In order to run the development environment, simply run the following command:
```
./run_env.sh
```
This spins up a container that listens on port 8080. If you go to `http://localhost:8080`, you'd enter the environment's portal.

### Reproducability

Each one of the branches in this repo represents an experiment. In order to re-produce experiments, simply checkout the branch and then run:
```
dvc pull
```
DVC will now try to fetch the relevant data for that specific branch.

### More info

For more information about the development environment, please check out this project:
https://github.com/ml-tooling/ml-workspace
