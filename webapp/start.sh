DOCKER=false
TAG="we-robot"

if DOCKER
then
  docker build -t $TAG .
  docker run $TAG
else
  flask run
fi
