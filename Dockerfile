FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /source/
COPY . /source/

RUN pip install -U pip \
    && pip install -r requirements.txt

ENV BERT_DIR=/bert
ENV DATA_DIR=/data
ENV OUTPUT_DIR=/output

CMD ["/script.sh"]