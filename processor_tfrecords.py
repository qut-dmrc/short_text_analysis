import datetime
import os
import threading
from pathlib import Path

import tensorflow as tf
from docopt import docopt

from bert_train import preprocess_df, convert_df_to_examples_mp, save_examples
from short_text_analysis.cloud_utils import read_df_gcs, setup_logging_local


def main():
    """ Convert JSON or CSV files to TFRecords to use with a BERT model

    Usage:
      process_tfrecords.py [-v] --config=config_file <gcs_input_path>

    Options:
      -h --help                 Show this screen.
      --config=config_file.py   The configuration file with model parameters, data path, etc
      -v --verbose              Enhanced logging
      --version  Show version.

    """
    args = docopt(main.__doc__, version='DMRC BERT preprocessor 0.1')

    import importlib.util
    spec = importlib.util.spec_from_file_location("classifier.config", args['--config'])
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    date_prefix = format(
        datetime.datetime.utcnow().strftime('%Y%m%d%H%M'))

    setup_logging_local(log_file_name=f'process_tfrecords_log_{date_prefix}.txt', verbose=args['--verbose'])

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cfg.PATH_TO_GOOGLE_KEY

    tf.logging.info('***** BERT pretrained directory: {} *****'.format(cfg.BERT_PRETRAINED_DIR))
    tf.logging.info(tf.gfile.ListDirectory(cfg.BERT_PRETRAINED_DIR))

    output_dir = str(Path(cfg.PREDICT_TFRECORDS).parent)
    tf.compat.v1.gfile.MakeDirs(output_dir)

    tf.logging.info('***** TFRecords output directory: {} *****'.format(output_dir))

    """ Convert all the input files to TensorFlow Records and save to GCS"""
    glob_list = tf.compat.v1.gfile.Glob(args['<gcs_input_path>'])

    t0 = datetime.datetime.now()

    for file in glob_list:
        t1 = datetime.datetime.now()
        stem = Path(file).stem
        gcs_output_file = os.path.join(output_dir, stem + f'_{cfg.BERT_MODEL}_{cfg.MAX_SEQUENCE_LENGTH}.tf_record')
        gcs_output_file_ids = gcs_output_file + '.ids.txt'

        existing_files = tf.compat.v1.gfile.ListDirectory(output_dir)
        if os.path.basename(gcs_output_file) in existing_files:
            tf.logging.info(f"Output file {gcs_output_file} already exists. Skipping input from {file}.")
            continue

        tf.logging.info(f"Reading from {file}.")

        df = read_df_gcs(file)

        if cfg.LABEL_FIELD not in df.columns:
            df[cfg.LABEL_FIELD] = None

        list_categories = cfg.CLASSIFICATION_CATEGORIES
        if not list_categories:
            list_categories = ['0']  # BERT's convert_single_example() needs at least one category, apparently.

        df = preprocess_df(df, id_field=cfg.ID_FIELD, label_field=None, list_of_all_fields=cfg.ALL_FIELDS,
                           list_of_text_fields=cfg.TEXT_FIELDS, drop_label=True)

        tf.logging.info(f"Tokenizing {df.shape[0]} rows from {file} in parallel.")
        tf_examples = convert_df_to_examples_mp(df, concurrency=cfg.CONCURRENCY, vocab_file=cfg.VOCAB_FILE,
                                                do_lower_case=True, label_list=list_categories,
                                                max_seq_length=cfg.MAX_SEQUENCE_LENGTH, is_predicting=True)

        # save examples asynchronously
        threading.Thread(target=save_examples, args=[tf_examples, gcs_output_file, gcs_output_file_ids]).start()
        tf.logging.info(f"Started saving features to {gcs_output_file}")

        tz = datetime.datetime.now()
        tf.logging.info(f"Finished file in: {tz - t1}")

    tz = datetime.datetime.now()
    tf.logging.info(f"Finished entire run in: {tz - t0}")


if __name__ == '__main__':
    main()
