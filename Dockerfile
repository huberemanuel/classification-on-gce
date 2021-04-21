FROM ubuntu:latest

LABEL maintainer="emanuel.tesv@gmail.com"

ENV APP_HOME /app
ENV PORT 8084
WORKDIR $APP_HOME

RUN apt-get update && apt-get upgrade -y

RUN apt-get install make python3.8 python3-pip -y

RUN pip3 install --upgrade pip

COPY . ./

RUN make install

CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 classifier.web.app:app 
