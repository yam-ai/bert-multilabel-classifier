import os
import csv
import collections
import tokenization
import tensorflow as tf


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, labels=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.labels = labels


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file):
        """Reads a comma separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MultiLabelTextProcessor(DataProcessor):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = None

    def get_train_examples(self):
        return self._get_examples("train.csv")

    def get_dev_examples(self):
        return self._get_examples("dev.csv")

    def get_test_examples(self):
        return self._get_examples("test.csv")

    def get_labels(self):
        """See base class."""
        with open(os.path.join(self.data_dir, "label.csv")) as f:
            reader = csv.reader(f)
            self.labels = next(reader)
        return self.labels

    def _get_examples(self, filename):
        return self._create_examples(self._read_csv(os.path.join(self.data_dir, filename)))

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = list()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            text = tokenization.convert_to_unicode(line[1])
            labels = line[2:]
            examples.append(InputExample(guid=guid, text=text, labels=labels))
        return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


def convert_single_example(ex_index, example, num_labels, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * num_labels,
            is_real_example=False)

    tokens_original = tokenizer.tokenize(example.text)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_original) > max_seq_length - 2:
        tokens_original = tokens_original[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_original:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if example.labels is not None:
        label_ids = [int(x) for x in example.labels]
    else:
        label_ids = [0] * num_labels
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: {} (id = {})".format(
            example.labels, label_ids))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        is_real_example=True)
    return feature


def convert_examples_to_features(examples, num_labels, max_seq_length, tokenizer):

    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
        return f

    example_string_list = list()
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, num_labels,
                                         max_seq_length, tokenizer)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        example_string_list.append(tf_example.SerializeToString())

    return example_string_list


def file_based_convert_examples_to_features(
        examples, num_labels, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)
    example_string_list = convert_examples_to_features(
        examples, num_labels, max_seq_length, tokenizer)
    for example_string in example_string_list:
        writer.write(example_string)
    writer.close()


def input_fn_builder(input_file, seq_length, num_labels, is_training,
                     drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([num_labels], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _cast_features(features):
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in features:
            t = features[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            features[name] = t
        return features

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        features = tf.parse_single_example(record, name_to_features)
        return _cast_features(features)

    def file_based_input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    def serving_input_receiver_fn():
        """An input_fn that expects a serialized tf.Example."""
        serialized_tf_example = tf.placeholder(
            dtype=tf.string,
            name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}

        features = tf.parse_example(serialized_tf_example, name_to_features)
        features = _cast_features(features)

        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    if input_file is not None:
        return file_based_input_fn
    else:
        return serving_input_receiver_fn
