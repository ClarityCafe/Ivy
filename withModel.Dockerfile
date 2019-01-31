FROM python:3.6-alpine

USER root

RUN apk add --no-cache --virtual .build-deps g++ libffi-dev openssl-dev && \
    pip install --upgrade pip setuptools && \
    mkdir -p /app && \
    chmod g+rw /app;

USER nobody

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt && mkdir -p resources/result

EXPOSE 4444

CMD ["/bin/sh", "run.sh"]