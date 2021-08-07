CONTAINER_ID=$(docker ps -aqf "name=ds-workspace")

if [[ ! -z "$CONTAINER_ID" ]]; then
  echo "Found an existing container with the same name. Deleting..."
  $(docker container rm -f $CONTAINER_ID)
fi

docker build -t "ds-workspace" .
docker run --env-file docker_env -d -v "/${PWD}:/workspace" --gpus all --name "ds-workspace" -p 8080:8080 --shm-size 2g --restart always "ds-workspace"

