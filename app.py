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

import falcon
from wsgiref import simple_server
from falcon.media.validators import jsonschema
import logging
import sys

import serve

request_schema = {
    'title': 'Label texts',
    'description': 'Predict labels to be assigned on texts',
    'type': 'object',
    'required': ['texts'],
    'properties': {
        'texts': {
            'type': 'array',
            'description': 'A list of texts for labeling',
            'items': {
                'type': 'object',
                'required': ['id', 'text'],
                'properties': {
                    'id': {
                        'type': 'integer',
                        'description': 'The id of the text'
                    },
                    'text': {
                        'type': 'string',
                        'description': 'A string of text',
                        'minLength': 1
                    }
                }
            }
        }
    }
}


class ClassifierResource:
    def __init__(self, logger, classifier):
        self.logger = logger
        self.classifier = classifier

    @jsonschema.validate(request_schema)
    def on_post(self, req, resp):
        """Handles POST requests"""
        texts = req.media.get("texts")
        try:
            result = self.classifier.predict(texts)
        except Exception as e:
            self.logger.error('Error occurred during prediction: {}'.format(e))
            raise falcon.HTTPInternalServerError(
                title='Internal server error',
                description='The service was unavailable. Please try again later.')

        resp.media = result


def create_app(model_dir):
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter(
            '[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s',
            '%Y-%m-%d %H:%M:%S %z')
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

    try:
        classifier = serve.MultiLabelClassifierServer(model_dir)
        app = falcon.API()
        app.add_route(
            '/classifier', ClassifierResource(logger=logger, classifier=classifier))
    except Exception as e:
        logger.error(
            'Failed to initialize with model directory {}: {}'.format(model_dir, e))
        sys.exit(4)  # tell gunicorn to exit

    logger.info('Serving multilabel classifier...')
    logger.info('Number of labels: {}'.format(classifier.num_labels))
    return app


if __name__ == "__main__":
    PORT = 8000
    MODEL_DIR = "/model"
    app = create_app(MODEL_DIR)
    with simple_server.make_server('', PORT, app) as httpd:
        print("Serving HTTP on port {}".format(PORT))
        httpd.serve_forever()
