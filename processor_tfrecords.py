import datetime
import os
import threading
from pathlib import Path

import tensorflow as tf
from docopt import docopt

from bert_train import preprocess_df, convert_df_to_examples_mp, save_examples
from cloud_utils import read_df_gcs, setup_logging_local


def main():
    """ Convert JSON or CSV files to TFRecords to use with a BERT model

    Usage:
      process_tfrecords.py --config=config_file <gcs_input_path>

    Options:
      -h --help                 Show this screen.
      --config=config_file.py   The configuration file with model parameters, data path, etc
      --version  Show version.

    """
    args = docopt(main.__doc__, version='DMRC BERT preprocessor 0.1')

    import importlib.util
    spec = importlib.util.spec_from_file_location("classifier.config", args['--config'])
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    date_prefix = format(
        datetime.datetime.utcnow().strftime('%Y%m%d%H%M'))

    setup_logging_local(log_file_name=f'process_tfrecords_log_{date_prefix}.txt')

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cfg.PATH_TO_GOOGLE_KEY

    tf.logging.info('***** BERT pretrained directory: {} *****'.format(cfg.BERT_PRETRAINED_DIR))
    tf.logging.info(tf.gfile.ListDirectory(cfg.BERT_PRETRAINED_DIR))

    tf.gfile.MakeDirs(cfg.PREDICT_TFRECORDS)
    tf.logging.info('***** TFRecords output directory: {} *****'.format(cfg.PREDICT_TFRECORDS))

    """ Convert all the input files to TensorFlow Records and save to GCS"""
    glob_list = tf.gfile.Glob(args['<gcs_input_path>'])

    t0 = datetime.datetime.now()

    for file in glob_list:
        t1 = datetime.datetime.now()
        stem = Path(file).stem
        gcs_output_file = os.path.join(cfg.PREDICT_TFRECORDS, stem + f'_{cfg.BERT_MODEL}_{cfg.MAX_SEQUENCE_LENGTH}.tf_record')
        gcs_output_file_ids = gcs_output_file + '.ids.txt'

        existing_files = tf.gfile.ListDirectory(cfg.PREDICT_TFRECORDS)
        if os.path.basename(gcs_output_file) in existing_files:
            tf.logging.info(f"Output file {gcs_output_file} already exists. Skipping input from {file}.")
            continue

        # should no longer need the line below, but keeping for now just in case the new method above doesn't work.
        if tf.gfile.Exists(gcs_output_file):
            tf.logging.warn(f"Output file {gcs_output_file} already exists. Skipping input from {file}.")
            continue

        tf.logging.info(f"Reading from {file}.")

        all_fields = (set(cfg.ALL_FIELDS) - set(cfg.LABEL_FIELD))

        df = read_df_gcs(file, [])
        df = preprocess_df(df, id_field=cfg.ID_FIELD, label_field=cfg.LABEL_FIELD, list_of_all_fields=all_fields,
                           list_of_text_fields=cfg.TEXT_FIELDS)

        tf.logging.info(f"Tokenizing {df.shape[0]} rows from {file} in parallel.")
        tf_examples = convert_df_to_examples_mp(df, concurrency=cfg.CONCURRENCY)

        # save examples asynchronously
        threading.Thread(target=save_examples, args=[tf_examples, gcs_output_file, gcs_output_file_ids]).start()
        tf.logging.info(f"Started saving features to {gcs_output_file}")

        tz = datetime.datetime.now()
        tf.logging.info(f"Finished file in: {tz - t1}")

    tz = datetime.datetime.now()
    tf.logging.info(f"Finished entire run in: {tz - t0}")


if __name__ == '__main__':
    main()
