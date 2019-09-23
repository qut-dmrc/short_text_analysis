"""
This module contains functions to train a simple multi-label classification processor

"""

import datetime
import functools
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from bert import run_classifier, modeling, optimization
from docopt import docopt
from sklearn.model_selection import train_test_split

import bert_classify_tfrc
from bert_utils import read_training_data_gcs, preprocess_df, convert_dataframe_to_features, \
    convert_dataframe_to_examples, save_examples
from cloud_utils import read_df_gcs, setup_logging_local, save_df_gcs


def main():
    """ Train a BERT model. Loads a pretrained model and finetunes it based on all
    training sets found in the specified directory.

    Outputs a new predicted dataset from live data that can be used for validation and further
    semi-supervised training.


    Usage:
      bert_train.py [--train] [--validate] --config=config_file [--tpu_name=name]

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

    tf.compat.v1.logging.info('***** Task data directory: {} *****'.format(cfg.TASK_DATA_DIR))
    tf.compat.v1.logging.info('***** BERT pretrained directory: {} *****'.format(cfg.BERT_PRETRAINED_DIR))
    assert cfg.BUCKET, 'Must specify an existing GCS bucket name'
    tf.compat.v1.logging.info('***** Model output directory: {} *****'.format(cfg.OUTPUT_DIR))
    tf.compat.v1.logging.info('***** Vocabulary file: {} *****'.format(cfg.VOCAB_FILE))
    tf.compat.v1.logging.info('***** Predictions directory: {} *****'.format(cfg.PREDICT_DIR))

    tpu_name = args['--tpu_name']

    if tpu_name:
        use_tpu = True
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_name)
        tpu_address = tpu_cluster_resolver.get_master()

        tf.compat.v1.logging.info("TPU address: {}".format(tpu_address))

        with tf.compat.v1.Session(cfg.TPU_ADDRESS) as session:
            # Upload credentials to TPU.
            if "COLAB_TPU_ADDR" in os.environ:
                tf.compat.v1.logging.info(f'TPU devices: {session.list_devices()}')

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

    tf.compat.v1.logging.info("Tensorflow version: {}".format(tf.__version__))

    tf.io.gfile.makedirs(cfg.OUTPUT_DIR)

    estimator = None

    if args['--train']:
        try:
            tf.compat.v1.logging.info("DELETING OUTPUT DIRECTORY: {}".format(cfg.OUTPUT_DIR))
            tf.io.gfile.rmtree(cfg.OUTPUT_DIR)
        except:
            # Doesn't matter if the directory didn't exist
            pass

        tf.io.gfile.makedirs(cfg.OUTPUT_DIR)

        # Train the model: Uses all training sets in the GCS bucket
        # - anything labeled 'training/*.csv'

        create_training_sets(cfg.TRAINING_SETS, cfg.ALL_FIELDS, cfg.LABEL_FIELD,
                            cfg.CLASSIFICATION_CATEGORIES, cfg.ID_FIELD,
                            cfg.TEXT_FIELDS, cfg.VOCAB_FILE, cfg.MAX_SEQUENCE_LENGTH,
                            cfg.DO_LOWER_CASE)

        label_list = cfg.CLASSIFICATION_CATEGORIES

        num_train_steps = int(
            len(training_examples) / cfg.TRAIN_BATCH_SIZE * cfg.NUM_TRAIN_EPOCHS)
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


def generate_random_validation(cfg, estimator):
    glob_list = tf.io.gfile.glob(cfg.PREDICT_INPUT_PATH)

    sample_file = random.choice(glob_list)
    stem = Path(sample_file).stem
    sample_file_tfrecords = os.path.join(cfg.PREDICT_TFRECORDS,
                                         stem + f'_{cfg.BERT_MODEL}_{cfg.MAX_SEQUENCE_LENGTH}.tf_record')

    tf.compat.v1.logging.info(f"Generating a validation dataset from a randomly chosen input file: {sample_file}")
    tf.compat.v1.logging.info(f"(Using TFRecords from : {sample_file_tfrecords})")

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

    gcs_output_path = cfg.PREDICT_DIR + '/' + run_date + '-randomsample-predicted-semisupervised-1k-each.csv'
    save_df_gcs(gcs_output_path, df_validate)
    tf.compat.v1.logging.info('Saved semi-supervised training set to: {}'.format(gcs_output_path))


def create_training_sets(training_sets, all_fields, label_field, classification_categories, id_field, text_fields,
                 vocab_file, max_sequence_length, do_lower_case):
    df = read_training_data_gcs(training_sets, all_fields, label_field, classification_categories)

    df = df[df[label_field].isin(classification_categories)]

    tf.compat.v1.logging.info('After filtering for categories, we are left with {} labeled rows'.format(df.shape[0]))
    tf.compat.v1.logging.info('Categories: {}'.format(classification_categories))

    tf.compat.v1.logging.info(df.info())

    tf.compat.v1.logging.info("Summary of labels:")
    tf.compat.v1.logging.info(df.groupby(by=label_field).agg({id_field: 'nunique'}))

    df = preprocess_df(df, list_of_all_fields=all_fields, list_of_text_fields=text_fields, label_field=label_field,
                       id_field=id_field)
    tf.compat.v1.logging.info(
        'After dropping null values and duplicates, we are left with {} labeled rows'.format(df.shape[0]))

    # 80 / 10 / 10 split train/val/test
    train_df, test_df = train_test_split(df, test_size=0.2)
    test_df, validation_df = train_test_split(train_df, test_size=0.5)

    train_guids_and_examples = convert_dataframe_to_examples(train_df, vocab_file, do_lower_case, classification_categories,
                             max_seq_length=max_sequence_length, is_predicting=False)
    test_guids_and_examples = convert_dataframe_to_examples(test_df, vocab_file, do_lower_case, classification_categories,
                             max_seq_length=max_sequence_length, is_predicting=False)
    validation_guids_and_examples = convert_dataframe_to_examples(validation_df, vocab_file, do_lower_case, classification_categories,
                             max_seq_length=max_sequence_length, is_predicting=False)

    save_examples(train_guids_and_examples, cfg.OUTPUT_DIR + '/train.tfrecords', cfg.OUTPUT_DIR + 'train.tfrecords.ids.txt')
    save_examples(test_guids_and_examples, cfg.OUTPUT_DIR + '/test.tfrecords', cfg.OUTPUT_DIR + 'test.tfrecords.ids.txt')
    save_examples(validation_guids_and_examples, cfg.OUTPUT_DIR + '/validate.tfrecords', cfg.OUTPUT_DIR + 'validate.tfrecords.ids.txt')



def get_loss_fn(num_classes, loss_factor=1.0):
  """Gets the classification loss function."""

  def classification_loss_fn(labels, logits):
    """Classification loss."""
    labels = tf.squeeze(labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(
        tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    loss *= loss_factor
    return loss

  return classification_loss_fn


def run_customized_training(strategy,
                            bert_config,
                            input_meta_data,
                            model_dir,
                            epochs,
                            steps_per_epoch,
                            steps_per_loop,
                            eval_steps,
                            warmup_steps,
                            initial_lr,
                            init_checkpoint,
                            use_remote_tpu=False,
                            custom_callbacks=None,
                            run_eagerly=False):
  """Run BERT classifier training using low-level API."""
  max_seq_length = input_meta_data['max_seq_length']
  num_classes = input_meta_data['num_labels']

  train_input_fn = functools.partial(
      input_pipeline.create_classifier_dataset,
      FLAGS.train_data_path,
      seq_length=max_seq_length,
      batch_size=FLAGS.train_batch_size)
  eval_input_fn = functools.partial(
      input_pipeline.create_classifier_dataset,
      FLAGS.eval_data_path,
      seq_length=max_seq_length,
      batch_size=FLAGS.eval_batch_size,
      is_training=False,
      drop_remainder=False)

  def _get_classifier_model():
    """Gets a classifier model."""
    classifier_model, core_model = (
        bert_models.classifier_model(bert_config, tf.float32, num_classes,
                                     max_seq_length))
    classifier_model.optimizer = optimization.create_optimizer(
        initial_lr, steps_per_epoch * epochs, warmup_steps)
    if FLAGS.fp16_implementation == 'graph_rewrite':
      # Note: when flags_obj.fp16_implementation == "graph_rewrite", dtype as
      # determined by flags_core.get_tf_dtype(flags_obj) would be 'float32'
      # which will ensure tf.compat.v2.keras.mixed_precision and
      # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
      # up.
      classifier_model.optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          classifier_model.optimizer)
    return classifier_model, core_model

  loss_fn = get_loss_fn(
      num_classes,
      loss_factor=1.0 /
      strategy.num_replicas_in_sync if FLAGS.scale_loss else 1.0)

  # Defines evaluation metrics function, which will create metrics in the
  # correct device and strategy scope.
  def metric_fn():
    return tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)

  return tf.models.official.nlp.bert.model_training_utils.run_customized_training_loop(
      strategy=strategy,
      model_fn=_get_classifier_model,
      loss_fn=loss_fn,
      model_dir=model_dir,
      steps_per_epoch=steps_per_epoch,
      steps_per_loop=steps_per_loop,
      epochs=epochs,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      eval_steps=eval_steps,
      init_checkpoint=init_checkpoint,
      metric_fn=metric_fn,
      use_remote_tpu=use_remote_tpu,
      custom_callbacks=custom_callbacks,
      run_eagerly=run_eagerly)


def export_classifier(model_export_path, input_meta_data):
  """Exports a trained model as a `SavedModel` for inference.
  Args:
    model_export_path: a string specifying the path to the SavedModel directory.
    input_meta_data: dictionary containing meta data about input and model.
  Raises:
    Export path is not specified, got an empty string or None.
  """
  if not model_export_path:
    raise ValueError('Export path is not specified: %s' % model_export_path)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  classifier_model = bert_models.classifier_model(
      bert_config, tf.float32, input_meta_data['num_labels'],
      input_meta_data['max_seq_length'])[0]
  model_saving_utils.export_bert_model(
      model_export_path, model=classifier_model, checkpoint_dir=FLAGS.model_dir)



if __name__ == '__main__':
    main()
