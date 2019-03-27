import tensorflow as tf
import os
import json
import time
import data
import tokenization


class MultiLabelClassifierServer(object):

    def __init__(self, saved_model_dir):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Load model configs
        self.vocab_file = os.path.join(
            saved_model_dir, "assets.extra", "vocab.txt")
        self.classifier_config_file = os.path.join(
            saved_model_dir, "assets.extra", "classifier_config.json")
        with open(classifier_config_file) as f:
            self.classifier_config = json.load(f)
        self.do_lower_case = classifier_config.get("do_lower_case")
        self.max_seq_length = classifier_config.get("max_seq_length")
        self.labels = classifier_config.get("labels")
        self.num_labels = len(labels)

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=self.do_lower_case)
        self.predictor = tf.contrib.predictor.from_saved_model(
            export_dir=saved_model_dir, config=config)

    def predict(self, query):
        examples = list()
        for i, text in enumerate(query):
            text = tokenization.convert_to_unicode(text)
            examples.append(data.InputExample(guid=i, text=text))

        example_string_list = data.convert_examples_to_features(
            examples, self.num_labels, self.max_seq_length, self.tokenizer)

        tic = time.time()
        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num examples = {}".format(len(example_string_list)))

        predictions = self.predictor({"examples": example_string_list})
        toc = time.time()
        tf.logging.info("Prediction time: {}s".format((toc - tic)))

        return predictions
