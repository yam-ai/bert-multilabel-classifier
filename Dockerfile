FROM tensorflow/tensorflow:1.12.0-py3

WORKDIR /source/
COPY . /source/

ENV BERT_DIR=/bert
ENV DATA_DIR=/data
ENV OUTPUT_DIR=/output

CMD ["/source/script.sh"]