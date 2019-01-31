FROM alpine:latest

USER root

RUN apk add --no-cache --virtual .build-deps g++ python3-dev libffi-dev openssl-dev && \
    apk add --no-cache --update python3 && \
    pip3 install --upgrade pip setuptools && \
    mkdir -p /app && \
    chmod g+rw /app;

COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 4444

CMD ["/bin/sh", "run.sh"]