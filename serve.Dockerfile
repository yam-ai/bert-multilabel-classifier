# coding=utf-8
# Copyright 2019 YAM AI Machinery Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

FROM tensorflow/tensorflow:1.15.4-py3

WORKDIR /source/
COPY . /source/

RUN pip install -U pip setuptools absl-py \
    && pip install falcon gunicorn jsonschema

ENV PORT=8000
ENV MODEL_DIR=/model
RUN echo '#!/usr/bin/env bash' > /serve.sh \
    && echo 'gunicorn -b 0.0.0.0:${PORT} "app:create_app(\"${MODEL_DIR}\")"' >> /serve.sh \
    && chmod +x /serve.sh

CMD ["/serve.sh"]
