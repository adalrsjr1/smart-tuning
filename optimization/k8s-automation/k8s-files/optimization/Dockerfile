FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY configsampler.py ./

ENTRYPOINT ["python", "configsampler.py"]