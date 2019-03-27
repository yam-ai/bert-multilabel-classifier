import falcon
from wsgiref import simple_server

import serve


class ClassifierResource:
    def __init__(self, classifier):
        self.classifier = classifier

    def on_post(self, req, resp):
        """Handles POST requests"""
        answers = self.classifier.predict(req.media)

        resp.media = answers


def create_app(model_dir):
    classifier = serve.MultiLabelClassifierServer(model_dir)

    app = falcon.API()
    app.add_route('/classifier', ClassifierResource(classifier=classifier))

    return app


if __name__ == "__main__":
    PORT = 8000
    MODEL_DIR = "/model"
    app = create_app(MODEL_DIR)
    with simple_server.make_server('', PORT, app) as httpd:
        print("Serving HTTP on port {}".format(PORT))
        httpd.serve_forever()
