# -*- coding: utf-8 -*-
"""Classify on TFRC
"""

import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from bert.run_classifier import file_based_input_fn_builder
from docopt import docopt

from bert_train import define_model
from cloud_utils import read_df_gcs, save_df_gcs, setup_logging_local


# Model Hyper Parameters

# Model configs

def main():
    """ Run predictions for an entire directory of tfrecords with a stored BERT model

    Usage:
      bert_classify_tfrc.py --config=config_file [--tpu_name=name]

    Options:
      -h --help                 Show this screen.
      --config=config_file.py   The configuration file with model parameters, data path, etc
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

    setup_logging_local(log_file_name=f'classify_{date_prefix}.txt')

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

    tf.logging.info("Tensorflow version: {}".format(tf.__version__))

    tf.gfile.MakeDirs(cfg.OUTPUT_DIR)

    estimator = define_model(cfg, tpu_address, use_tpu)

    predict_all_in_dir(cfg, estimator)


def predict_all_in_dir(cfg, estimator):
    """# Run predictions on all files"""
    import os
    tf.logging.info('***** Records to predict: {} *****'.format(cfg.PREDICT_SOURCE_RECORDS))
    tf.logging.info('***** Predictions save directory: {} *****'.format(cfg.PREDICT_OUTPUT_DIR))
    # quieten tensorflow for the prediction run
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    t0 = datetime.datetime.now()
    tf.logging.info('***** Started predictions at {} *****'.format(t0))
    tf.logging.set_verbosity(tf.logging.WARN)
    for predict_file in tf.gfile.Glob(cfg.PREDICT_SOURCE_TFRECORDS):
        stem = Path(predict_file).stem
        predict_output_file_merged = os.path.join(cfg.PREDICT_OUTPUT_DIR, stem + '.merged.csv')
        predict_output_file_lock = os.path.join(cfg.PREDICT_OUTPUT_DIR, stem + '.LOCK')

        if tf.gfile.Exists(predict_output_file_merged) or tf.gfile.Exists(predict_output_file_lock):
            tf.logging.warn(
                "Output file {} already exists. Skipping input from {}.".format(predict_output_file_merged,
                                                                                predict_file))
            continue
        with tf.gfile.Open(predict_output_file_lock, mode="w") as f:
            f.write('Locked at {}'.format(t0))

        df_merged = predict_single_file(cfg, estimator, predict_file)
        save_df_gcs(predict_output_file_merged, df_merged)
        tf.gfile.Remove(predict_output_file_lock)

    tz = datetime.datetime.now()
    tf.logging.warn('***** Finished all predictions at {}; {} total time *****'.format(tz, tz - t0))
    tf.logging.set_verbosity(tf.logging.INFO)


def predict_single_file(cfg, estimator, predict_file):
    t1 = datetime.datetime.now()
    tf.logging.warn("Predicting from {}.".format(predict_file))
    # Warning: According to tpu_estimator.py Prediction on TPU is an
    # experimental feature and hence not supported here
    #  raise ValueError("Prediction in TPU not supported")
    predict_drop_remainder = True
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=cfg.MAX_SEQUENCE_LENGTH,
        is_training=False,
        drop_remainder=True)
    results = []
    for prediction in estimator.predict(input_fn=predict_input_fn):
        results.append({'predicted_class': np.argmax(prediction['probabilities']),
                        'predicted_class_label': cfg.CLASSIFICATION_CATEGORIES[
                            np.argmax(prediction['probabilities'])],
                        'confidence': prediction['probabilities'][np.argmax(prediction['probabilities'])]})
    # Here results are stored as an ordered list - need to get the ID back from the ids.txt file.
    tz = datetime.datetime.now()
    tf.logging.warn('***** Finished predictions at {}; {} file time *****'.format(tz, tz - t1))
    predict_file_ids = predict_file + '.ids.txt'
    df_ids = read_df_gcs(predict_file_ids, header_rows=None)
    df_ids.columns = ['guid']
    df_results = pd.DataFrame(results)
    df_merged = pd.concat([df_ids, df_results], axis=1)
    df_merged = df_merged.rename(columns={'guid': cfg.ID_FIELD})

    return df_merged


if __name__ == '__main__':
    main()
