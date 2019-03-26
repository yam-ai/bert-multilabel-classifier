import tensorflow as tf
import os
import time
import data
import tokenization


class MultiLabelClassifierServer(object):

    def load_saved_model(self, saved_model_dir, do_lower_case=False):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        vocab_file = os.path.join(saved_model_dir, "assets.extra", "vocab.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.predictor = tf.contrib.predictor.from_saved_model(
            export_dir=saved_model_dir, config=config)
        self.num_labels = 12
        self.max_seq_length = 512

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
