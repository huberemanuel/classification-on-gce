FROM ubuntu:latest

LABEL maintainer="emanuel.tesv@gmail.com"

RUN apt-get update && apt-get upgrade -y

RUN apt-get install make python3.8 python3-pip -y

RUN pip3 install --upgrade pip

COPY . /app

RUN cd /app && make install

RUN cd /app && make train