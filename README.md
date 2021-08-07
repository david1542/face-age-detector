## ds-boilerplate

This repository serves as an initial boilerplate for a data science project. It's built on top of `ml-workspace`, with some setup.

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

### More info

For more information about the development environment, please check out this project:
https://github.com/ml-tooling/ml-workspace
