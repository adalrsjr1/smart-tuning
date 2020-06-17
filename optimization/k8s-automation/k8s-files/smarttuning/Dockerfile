# image-name smarttuning
FROM python:3.8-slim-buster
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY requirements.txt ./
COPY tests/letter-recognition.data ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get remove -y build-essential && apt-get autoremove -y

COPY . .

ENTRYPOINT ["python", "app.py"]