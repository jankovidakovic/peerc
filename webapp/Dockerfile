# 1. Build application
#============================================================
FROM python:3.9

# file structure
WORKDIR /python-docker
EXPOSE 5000

RUN pip install -U pip

# install dependencies
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# specify flask entrypoint directory
ENV FLASK_APP=myapp.py

# Copy source files
COPY . .

# 2. Run

# flag --host exposes docker container to outside requests
CMD [ "flask", "run", "--host=0.0.0.0"]
