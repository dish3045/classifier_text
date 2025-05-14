FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel


WORKDIR /app

RUN apt-get update && apt-get install -y git && apt-get clean
RUN apt-get install -y protobuf-compiler

RUN pip install --upgrade pip
RUN pip install transformers protobuf sentencepiece


COPY . .

# Force transformers and torch to use CPU by default
ENV CUDA_VISIBLE_DEVICES=""

CMD ["python", "main.py"]