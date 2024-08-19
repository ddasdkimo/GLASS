# 使用官方 Python 3.8 镜像作為基底
FROM python:3.9-slim


# 設定工作目錄
WORKDIR /app


# 將當前目錄下的所有文件複製到容器中
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install libgl1-mesa-glx libglib2.0-0 -y
# docker build -t glass .

# docker run --rm -itd --gpus all --shm-size=50g -v `pwd`:/app   -w /app --name glass glass
