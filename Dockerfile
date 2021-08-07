FROM mltooling/ml-workspace-gpu:latest

COPY requirements.txt /workspace/requirements.txt
COPY docker_env /workspace/docker_env

WORKDIR /workspace

RUN pip uninstall -y enum34
RUN pip install --ignore-installed -r requirements.txt
