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
        with open(self.classifier_config_file) as f:
            self.classifier_config = json.load(f)
        self.do_lower_case = self.classifier_config.get("do_lower_case")
        self.max_seq_length = self.classifier_config.get("max_seq_length")
        self.labels = self.classifier_config.get("labels")
        self.num_labels = len(self.labels)

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)
        self.predictor = tf.contrib.predictor.from_saved_model(
            export_dir=saved_model_dir, config=config)

    def predict(self, texts):
        examples = list()
        for item in texts:
            id = item.get("id")
            text = item.get("text")
            text = tokenization.convert_to_unicode(text)
            examples.append(data.InputExample(guid=id, text=text))

        example_string_list = data.convert_examples_to_features(
            examples, self.num_labels, self.max_seq_length, self.tokenizer)

        tic = time.time()
        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num examples = {}".format(len(example_string_list)))

        predictions = self.predictor({"examples": example_string_list})
        scores = predictions.get("probabilities").tolist()
        toc = time.time()
        tf.logging.info("Prediction time: {}s".format((toc - tic)))

        for i, item in enumerate(texts):
            item["scores"] = dict(zip(self.labels, scores[i]))
        return texts
