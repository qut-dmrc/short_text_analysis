"""
This module contains functions to train a simple multi-label classification processor

"""

import collections
import datetime
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from bert import run_classifier, modeling, optimization, tokenization
from docopt import docopt
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.model_selection import train_test_split
from tensorflow.python.lib.io import file_io

import bert_classify_tfrc
from cloud_utils import read_df_gcs, setup_logging_local, save_df_gcs


def main():
    """ Train a BERT model. Loads a pretrained model and finetunes it based on all
    training sets found in the specified directory.

    Outputs a new predicted dataset from live data that can be used for validation and further
    semi-supervised training.


    Usage:
      bert_train.py --train --config=config_file [--tpu_name=name]
      bert_train.py --validate --config=config_file [--tpu_name=name]

    Options:
      -h --help                 Show this screen.
      --config=config_file.py   The configuration file with model parameters, data path, etc
      --train                   Run fine-tuning from training datasets - Beware, this deletes
                                any existing fine-tuned model
      --validate                Generate a validation dataset from real data
      --tpu_name=name           The name of the TPU or cluster to run on
      --version  Show version.

    """

    args = docopt(main.__doc__, version='DMRC BERT Classifier 0.1')

    import importlib.util
    spec = importlib.util.spec_from_file_location("classifier.config", args['--config'])
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    date_prefix = format(
        datetime.datetime.utcnow().strftime('%Y%m%d%H%M'))

    setup_logging_local(log_file_name=f'train_{date_prefix}.txt')

    tf.logging.info('***** Task data directory: {} *****'.format(cfg.TASK_DATA_DIR))
    tf.logging.info('***** BERT pretrained directory: {} *****'.format(cfg.BERT_PRETRAINED_DIR))
    assert cfg.BUCKET, 'Must specify an existing GCS bucket name'
    tf.logging.info('***** Model output directory: {} *****'.format(cfg.OUTPUT_DIR))
    tf.logging.info('***** Vocabulary file: {} *****'.format(cfg.VOCAB_FILE))
    tf.logging.info('***** Predictions directory: {} *****'.format(cfg.PREDICT_DIR))

    tpu_name = args['--tpu_name']

    if tpu_name:
        use_tpu = True
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_name)
        tpu_address = tpu_cluster_resolver.get_master()

        tf.logging.info("TPU address: {}".format(tpu_address))

        with tf.Session(cfg.TPU_ADDRESS) as session:
            # Upload credentials to TPU.
            if "COLAB_TPU_ADDR" in os.environ:
                tf.logging.info(f'TPU devices: {session.list_devices()}')

                with open('/content/adc.json', 'r') as f:
                    auth_info = json.load(f)
                tf.contrib.cloud.configure_gcs(session, credentials=auth_info)
    else:
        use_tpu = False
        tpu_address = None

        # auth to Google

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cfg.PATH_TO_GOOGLE_KEY

        import subprocess
        cfg.num_gpu_cores = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')

    tf.logging.info("Tensorflow version: {}".format(tf.__version__))

    tf.gfile.MakeDirs(cfg.OUTPUT_DIR)

    estimator = None

    if args['--train']:
        try:
            tf.logging.info("DELETING OUTPUT DIRECTORY: {}".format(cfg.OUTPUT_DIR))
            tf.gfile.DeleteRecursively(cfg.OUTPUT_DIR)
        except:
            # Doesn't matter if the directory didn't exist
            pass

        tf.gfile.MakeDirs(cfg.OUTPUT_DIR)

        # Train the model: Uses all training sets in the GCS bucket
        # - anything labeled 'training/*.csv'

        processor = ClassificationTrainingProcessor(cfg.TRAINING_SETS, cfg.ALL_FIELDS, cfg.LABEL_FIELD,
                                                    cfg.CLASSIFICATION_CATEGORIES, cfg.ID_FIELD,
                                                    cfg.TEXT_FIELDS, cfg.VOCAB_FILE, cfg.MAX_SEQUENCE_LENGTH,
                                                    cfg.DO_LOWER_CASE)

        label_list = processor.get_labels()
        train_examples = processor.get_train_examples()

        num_train_steps = int(
            len(train_examples) / cfg.TRAIN_BATCH_SIZE * cfg.NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * cfg.WARMUP_PROPORTION)

        estimator = define_model(cfg, tpu_address, use_tpu, num_train_steps, num_warmup_steps)

        train_classifier(cfg, estimator, num_train_steps, train_examples)

        eval_classifier(cfg, estimator, processor, use_tpu)

        test_classifier(cfg, estimator, label_list, processor)

    if args['--validate']:
        # Load the stored model if we have one and didn't just train it.
        if not args['--train']:
            estimator = define_model(cfg, tpu_address, use_tpu)

        assert estimator is not None

        ## Predict on live data and output a sample for validation or training
        generate_random_validation(cfg, estimator)


def test_classifier(cfg, estimator, label_list, processor):
    # Run predictions on test data
    test_examples = processor.get_test_examples()
    t0 = datetime.datetime.now()
    print('***** Started test predictions at {} *****'.format(t0))
    tf.logging.info("  Num examples = %d", len(test_examples))
    tf.logging.info("  Batch size = %d", cfg.PREDICT_BATCH_SIZE)

    tf.logging.warn("Prediction on TPU is not supported. Some records will be dropped.")
    test_drop_remainder = True
    test_input_fn = run_classifier.input_fn_builder(
        features=test_examples,
        seq_length=cfg.MAX_SEQUENCE_LENGTH,
        is_training=False,
        drop_remainder=test_drop_remainder)
    result = estimator.predict(input_fn=test_input_fn)
    t1 = datetime.datetime.now()
    output_test_file = os.path.join(cfg.OUTPUT_DIR, "test_results.tsv")
    results = []
    with tf.gfile.GFile(output_test_file, "w") as writer:
        tf.logging.info("***** Test results *****")
        for prediction in result:
            results.append({'predicted_class': np.argmax(prediction['probabilities']),
                            'predicted_class_label': label_list[np.argmax(prediction['probabilities'])],
                            'confidence': prediction['probabilities'][np.argmax(prediction['probabilities'])]})
            output_line = "\t".join(
                str(class_probability) for class_probability in prediction) + "\n"
            writer.write(output_line)
    print('***** Finished test predictions at {}; {} total time *****'.format(t1, t1 - t0))
    predictions = np.array([p['predicted_class'] for p in results])
    label_ids = np.array([x.label_id for x in test_examples])
    if len(predictions) < len(label_ids):
        label_ids = label_ids[:len(predictions)]
    confusion_matrix = tf.confusion_matrix(label_ids, predictions, num_classes=len(label_list), name="tests")
    with tf.Session():
        tf.logging.info(
            'Confusion Matrix: \n\n {}'.format(tf.Tensor.eval(confusion_matrix, feed_dict=None, session=None)))
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    y_pred = predictions
    y_true = label_ids
    tf.logging.info("accuracy: {}".format(accuracy_score(y_true, y_pred)))
    for av in ['micro', 'macro', 'weighted']:
        tf.logging.info("f1 ({}): {}".format(av, f1_score(y_true, y_pred, average=av)))
        tf.logging.info("recall ({}): {}".format(av, recall_score(y_true, y_pred, average=av)))
        tf.logging.info("precision ({}): {}".format(av, precision_score(y_true, y_pred, average=av)))


def eval_classifier(cfg, estimator, processor, use_tpu):
    # Eval the model.
    eval_examples = processor.get_dev_examples()
    t0 = datetime.datetime.now()
    tf.logging.info('***** Started evaluation at {} *****'.format(t0))
    tf.logging.info('  Num examples = {}'.format(len(eval_examples)))
    tf.logging.info('  Batch size = {}'.format(cfg.EVAL_BATCH_SIZE))
    if use_tpu:
        tf.logging.warn("Eval will be slightly WRONG on the TPU because it will truncate the last batch.")
    eval_steps = int(len(eval_examples) / cfg.EVAL_BATCH_SIZE)
    eval_input_fn = run_classifier.input_fn_builder(
        features=eval_examples,
        seq_length=cfg.MAX_SEQUENCE_LENGTH,
        is_training=False,
        drop_remainder=True)
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    t1 = datetime.datetime.now()
    tf.logging.info('***** Finished evaluation at {}; {} total time *****'.format(t1, t1 - t0))
    output_eval_file = os.path.join(cfg.OUTPUT_DIR, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            tf.logging.info('  {} = {}'.format(key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))


def train_classifier(cfg, estimator, num_train_steps, train_examples):
    t0 = datetime.datetime.now()
    tf.logging.info('Training on BERT base model. This usually takes < 5 minutes. Please wait...')
    tf.logging.info('***** Started training at {} *****'.format(t0))
    tf.logging.info('  Num examples = {}'.format(len(train_examples)))
    tf.logging.info('  Batch size = {}'.format(cfg.TRAIN_BATCH_SIZE))
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = run_classifier.input_fn_builder(
        features=train_examples,
        seq_length=cfg.MAX_SEQUENCE_LENGTH,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    t1 = datetime.datetime.now()
    tf.logging.info('***** Finished training at {}; {} total time *****'.format(t1, t1 - t0))


def generate_random_validation(cfg, estimator):
    glob_list = tf.gfile.Glob(cfg.PREDICT_INPUT_PATH)

    sample_file = random.choice(glob_list)
    stem = Path(sample_file).stem
    sample_file_tfrecords = os.path.join(cfg.PREDICT_TFRECORDS,
                                         stem + f'_{cfg.BERT_MODEL}_{cfg.MAX_SEQUENCE_LENGTH}.tf_record')

    tf.logging.info(f"Generating a validation dataset from a randomly chosen input file: {sample_file}")
    tf.logging.info(f"(Using TFRecords from : {sample_file_tfrecords})")

    df = bert_classify_tfrc.predict_single_file(cfg, estimator, sample_file_tfrecords)

    df_sample = read_df_gcs(sample_file)
    df = pd.merge(df_sample, df, left_on=cfg.ID_FIELD, right_on=cfg.ID_FIELD)

    # Get 1000 rows from each class
    df_predictions = df.groupby('predicted_class').apply(lambda s: s.sample(min(len(s), 1000)))
    df_predictions = df_predictions.reset_index(drop=True)
    df_predictions["DATA_SOURCE"] = "{} random from class".format(stem)

    # get 1000 least confident results for active labeling
    df_least_confident = df.sort_values(by=['confidence'], ascending=True)[:1000]
    df_least_confident["DATA_SOURCE"] = "{} least confident".format(stem)

    df_validate = pd.concat([df_predictions, df_least_confident])
    df_validate = df_validate.drop_duplicates(subset=cfg.ID_FIELD)

    df_validate[cfg.LABEL_FIELD] = None

    run_date = datetime.datetime.strftime(datetime.datetime.utcnow(), '%Y%m%d%H%M')

    gcs_output_path = cfg.OUTPUT_DIR + '/' + run_date + '-randomsample-predicted-semisupervised-1k-each.csv'
    save_df_gcs(gcs_output_path, df_validate)
    tf.logging.info('Saved semi-supervised training set to: {}'.format(gcs_output_path))


def convert_numeric_label_to_string(x, list_of_categories):
    try:
        x = list_of_categories[int(x)]
        return x
    except ValueError:
        return x
    except IndexError:
        return None


def read_training_data_gcs(gcs_path, list_of_all_fields, label_field, list_of_categories):
    df = pd.DataFrame()
    dataframes = []
    for train_file in tf.gfile.Glob(gcs_path):
        data = read_df_gcs(train_file, list_of_all_fields)
        tf.logging.info('Found columns: {}'.format(data.columns))
        tf.logging.info('Found categories: {}'.format(data[label_field].unique()))
        for field in list_of_all_fields:
            assert field in data.columns, "Column {} not found in dataset {}!".format(field, gcs_path)
        assert data.shape[0] > 0, "No data found in dataset!"

        # convert numeric labels to strings
        if is_numeric_dtype(data[label_field]):
            data[label_field] = data[label_field].apply(convert_numeric_label_to_string,
                                                        list_of_categories=list_of_categories)

        dataframes.append(data)

    if dataframes:
        df = pd.concat(dataframes)

    assert df.shape[0] > 0, "No data found!"

    tf.logging.info("Found {} rows total.".format(df.shape[0]))

    return df


def preprocess_df(df, list_of_all_fields, list_of_text_fields, label_field, id_field):
    """ Pre-process the dataframe - wrangle the columns into a standard format."""
    tf.logging.info('Starting to preprocess dataframe containing {} rows.'.format(df.shape[0]))

    assert all(
        elem in df.columns for elem in
        list_of_all_fields), "Dataset appears to be missing columns - please check your inputs."

    # If we have multiple rows for one text, drop the later ones.
    df = df.drop_duplicates(subset=list_of_text_fields)

    # combine multiple text fields if neccessary
    if len(list_of_text_fields) > 1:
        df['text_a'] = df[list_of_text_fields[0]].str.cat(df[list_of_text_fields[1:]], sep=' ', na_rep='')
    else:
        df['text_a'] = df[list_of_text_fields[0]]

    if label_field:
        df['label'] = df[label_field]
    else:
        df['label'] = ""

    if is_string_dtype(df[label_field]):
        df[label_field] = df[label_field].str.lower()

    df['guid'] = df[id_field]

    df = df[['guid', 'text_a', 'label']]

    # drop fields with empty values
    df = df.dropna(axis='index', subset=['guid', 'text_a'])
    if label_field:  # if we were reading labeled data, only keep labeled records
        df = df.dropna(axis='index', subset=['label'])

    for col in df.columns:
        # convert to unicode
        df[col] = df[col].astype('unicode')

    return df


def convert_df_to_examples_mp(df, concurrency):
    batch_size = int(math.ceil(df.shape[0] / concurrency))
    list_df = [df[i:i + batch_size] for i in range(0, df.shape[0], batch_size)]
    tf.logging.info('Chunked dataframe into {} chunks.'.format(len(list_df)))

    with mp.Pool(processes=concurrency) as pool:
        results = pool.map(convert_dataframe_to_examples, list_df)

    flattened_list_of_examples = list(itertools.chain(*results))
    tf.logging.info('Finished processing dataframe.')

    return flattened_list_of_examples


def replace_special_tokens(text):
    text = re.sub(r"\B#\b", "xxhashtagxx ", text)
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "xxurlxx ", text)
    text = re.sub(r"\B@\w{1,15}\b", "xxusernamexx ", text)
    text = re.sub(r'(\"@|\brt @|\bmt @|\bvia @)(\w+)', "xxretweet ", text)

    return text


def convert_dataframe_to_examples(df, vocab_file, do_lower_case, label_list, max_seq_length, is_predicting=False):
    """Creates a list of TFRecords from the dataframe rows."""
    t0 = datetime.datetime.utcnow()
    examples_and_guids = []

    list_guids_and_features = convert_dataframe_to_features(df, vocab_file, do_lower_case, label_list, max_seq_length,
                                                            is_predicting)

    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    for guid, feature in list_guids_and_features:
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        examples_and_guids.append((guid, tf_example))

    time_taken = datetime.datetime.utcnow() - t0
    tf.logging.info("Finished converting dataframe chunk in {}.".format(time_taken))

    return examples_and_guids


def convert_dataframe_to_features(df, vocab_file, do_lower_case, label_list, max_seq_length, is_predicting=False):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    list_guids_and_features = []
    for index, row in df.iterrows():
        guid = tokenization.convert_to_unicode(row['guid'])
        text_a = tokenization.convert_to_unicode(row['text_a'])
        if is_predicting:
            label = '0'
        else:
            label = tokenization.convert_to_unicode(row['label'])

        text_a = replace_special_tokens(text_a)

        example = (
            run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        feature = run_classifier.convert_single_example(index, example, label_list,
                                                        max_seq_length, tokenizer)
        list_guids_and_features.append((guid, feature))
    return list_guids_and_features


def save_examples(list_of_examples, output_file_examples, output_file_guids):
    tf.logging.info("Saving {} to {} and {}.".format(len(list_of_examples), output_file_examples, output_file_guids))
    example_writer = tf.python_io.TFRecordWriter(output_file_examples)
    guid_writer = file_io.FileIO(output_file_guids, mode='w')

    for guid, example in list_of_examples:
        example_writer.write(example.SerializeToString())
        guid_writer.write(guid + "\n")
    guid_writer.close()
    example_writer.close()
    tf.logging.info("Done saving.")


class ClassificationTrainingProcessor():
    """ A simple multi-label classification processor"""

    def __init__(self, training_sets, all_fields, label_field, classification_categories, id_field, text_fields,
                 vocab_file, max_sequence_length, do_lower_case):
        self.classification_categories = classification_categories
        self.max_sequence_length = max_sequence_length
        self.vocab_file = vocab_file
        self.do_lower_case = do_lower_case
        df = read_training_data_gcs(training_sets, all_fields, label_field, classification_categories)

        df = df[df[label_field].isin(classification_categories)]

        tf.logging.info('After filtering for categories, we are left with {} labeled rows'.format(df.shape[0]))
        tf.logging.info('Categories: {}'.format(classification_categories))

        tf.logging.info(df.info())

        tf.logging.info("Summary of labels:")
        tf.logging.info(df.groupby(by=label_field).agg({id_field: 'nunique'}))

        df = preprocess_df(df, list_of_all_fields=all_fields, list_of_text_fields=text_fields, label_field=label_field,
                           id_field=id_field)
        tf.logging.info(
            'After dropping null values and duplicates, we are left with {} labeled rows'.format(df.shape[0]))

        # 70 / 20 / 10 split train/val/test
        self.train_df, self.test_df = train_test_split(df, test_size=0.1)
        self.train_df, self.validation_df = train_test_split(self.train_df, test_size=0.22)

    def get_train_examples(self):
        return self._create_examples(self.train_df, "train")

    def get_dev_examples(self):
        return self._create_examples(self.validation_df, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test_df, "test")

    def get_labels(self):
        return self.classification_categories

    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if set_type == 'predict':
            examples_and_guids = convert_dataframe_to_features(df, vocab_file=self.vocab_file,
                                                               do_lower_case=self.do_lower_case,
                                                               label_list=self.classification_categories,
                                                               max_seq_length=self.max_sequence_length,
                                                               is_predicting=True)
        else:
            examples_and_guids = convert_dataframe_to_features(df, vocab_file=self.vocab_file,
                                                               do_lower_case=self.do_lower_case,
                                                               label_list=self.classification_categories,
                                                               max_seq_length=self.max_sequence_length,
                                                               is_predicting=False)

        examples = [x for g, x in examples_and_guids]
        return examples


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # This classifier works on a segment level
    # If you want to use the token-level output, use model.get_sequence_output()
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    # Create our own output layer
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        # predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))

        # compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                mean_per_class_accuracy = tf.metrics.mean_per_class_accuracy(
                    label_ids,
                    predictions,
                    num_classes=num_labels,
                    weights=is_real_example
                )
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    "mean_per_class_accuracy": mean_per_class_accuracy,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def define_model(cfg, tpu_address, use_tpu, num_train_steps=-1, num_warmup_steps=-1, new_model=False):

    if use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address)

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=cfg.OUTPUT_DIR,
            save_checkpoints_steps=cfg.SAVE_CHECKPOINTS_STEPS,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=cfg.ITERATIONS_PER_LOOP,
                num_shards=cfg.NUM_TPU_CORES,
                per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))
    else:
        if cfg.num_gpu_cores >= 2:
            ### This doesn't work yet -- we only have one data file for all the workers and no way to split it.
            ### See: https://www.tensorflow.org/api_docs/python/tf/contrib/distribute/MirroredStrategy
            ### "Note that there has to be at least one input file per worker. If you have less than one input file
            ### per worker, we suggest that you should disable distributing your dataset using the method below."
            dist_strategy = tf.distribute.MirroredStrategy(num_gpus=cfg.num_gpu_cores)

            # tf.contrib.distribute.MirroredStrategy(
            #                num_gpus=cfg.num_gpu_cores,
            #                cross_device_ops=AllReduceCrossDeviceOps('nccl', num_packs=cfg.num_gpu_cores),
            #            )
            tf.logging.info(f"Running on {cfg.num_gpu_cores} GPU cores using Mirrored strategy.")
        else:
            dist_strategy = None
            
        run_config = tf.contrib.tpu.RunConfig(
            model_dir=cfg.OUTPUT_DIR,
            save_checkpoints_steps=cfg.SAVE_CHECKPOINTS_STEPS,
            train_distribute=dist_strategy,
            eval_distribute=dist_strategy,
        )

    if new_model:
        init_checkpoint = cfg.BERT_PRETRAINED_DIR + '/model.ckpt'
    else:
        init_checkpoint = tf.train.latest_checkpoint(cfg.OUTPUT_DIR)

    model_fn = model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(cfg.CONFIG_FILE),
        num_labels=len(cfg.CLASSIFICATION_CATEGORIES),
        init_checkpoint=init_checkpoint,
        learning_rate=cfg.LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=use_tpu,
        use_one_hot_embeddings=True)
    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=cfg.TRAIN_BATCH_SIZE,
        eval_batch_size=cfg.EVAL_BATCH_SIZE,
        predict_batch_size=cfg.PREDICT_BATCH_SIZE)
    return estimator


if __name__ == '__main__':
    main()
