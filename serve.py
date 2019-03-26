import tensorflow as tf
import time
import data


class MultiLabelClassifierServer(object):

    def load_saved_model(self, saved_model_dir):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.predictor = tf.contrib.predictor.from_saved_model(
            export_dir=saved_model_dir, config=config)
        self.num_labels = 12
        self.max_seq_length = 256

    def predict(query):
        examples = list()
        for i, text in enumerate(query):
            text = tokenization.convert_to_unicode(text)
            examples.append(data.InputExample(guid=i, text=text))

        example_string_list = data.convert_examples_to_features(
            examples, self.num_labels, self.max_seq_length)

        tic = time.time()
        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num examples = {}".format(len(example_string_list)))

        predictions = self.predictor({"examples": example_string_list})
        toc = time.time()
        tf.logging.info("Prediction time: {}s".format((toc - tic)))

        return predictions
