FROM tensorflow/tensorflow:1.12.0-py3

WORKDIR /source/
COPY . /source/

RUN pip install -U pip \
    && pip install falcon gunicorn

ENV PORT=8000
ENV MODEL_DIR=/model
RUN echo '#!/usr/bin/env bash' > /serve.sh \
    && echo 'gunicorn -b 0.0.0.0:${PORT} "app:create_app(\"${MODEL_DIR}\")"' >> /serve.sh \
    && chmod +x /serve.sh

CMD ["/serve.sh"]