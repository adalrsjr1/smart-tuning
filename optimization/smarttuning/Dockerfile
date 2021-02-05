# FROM python:3.9.1-alpine3.12 AS builder
# ENV PATH=$PATH:/root/.local/bin
# RUN /usr/local/bin/python -m pip install --upgrade pip
# RUN apk --no-cache --update-cache add gcc gfortran build-base wget freetype-dev libpng-dev openblas-dev
# RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
# COPY requirements.txt .
# RUN pip wheel --no-cache-dir --wheel-dir=/root/wheels -r requirements.txt
# 
# FROM python:3.9.1-alpine3.12
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONIOENCODING=UTF-8 \
#     PYTHONHASHSEED=31 \
#     PATH=$PATH:/root/.local/bin
# 
# RUN apk --no-cache --update-cache add openblas libstdc++
# WORKDIR /code
# 
# COPY --from=builder /root/wheels /root/wheels
# COPY --from=builder requirements.txt .
# 
# # Ignore the Python package index
# # and look for archives in
# # /root/wheels directory
# RUN pip install \
#       --no-cache-dir \
#       --no-index \
#       --find-links=/root/wheels \
#       -r requirements.txt
# COPY . .
# 
# ENTRYPOINT ["python", "./app.py"]



# image-name smarttuning
FROM python:3.9.1-slim-buster
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PYTHONHASHSEED=31
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY requirements.txt ./
COPY tests/letter-recognition.data ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get remove -y build-essential && apt-get autoremove --purge -y && apt-get clean -y

COPY . .

ENTRYPOINT ["python", "app.py"]
