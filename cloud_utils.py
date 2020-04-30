import csv
import logging
import os
from io import StringIO, BytesIO

import pandas as pd
import tensorflow as tf
from tensorflow.python.lib.io import file_io


def save_df_gcs(gcs_path, df, save_csv=True):
    tf.logging.info('saving dataframe to GCS: '.format(gcs_path))
    with file_io.FileIO(gcs_path, mode='w') as file_stream:
        if save_csv:
            df.to_csv(file_stream, encoding='utf-8', quoting=csv.QUOTE_ALL)
        else:
            df.to_json(file_stream, encoding='utf-8', orient='records', lines=True)


def read_df_gcs(gcs_path, list_of_all_fields=None, header_rows=0):
    """ Read the input data from Google Cloud Storage
    :param gcs_path: a single input file
    :param list_of_all_fields: all of the fields to read from the file (only if CSV)
    """
    tf.logging.info('downloading file from {}'.format(gcs_path))

    gzip = (gcs_path[-2:] == 'gz')
    json = (gcs_path[-4:] == 'json' or gcs_path[-7:] == 'json.gz')

    if gzip:
        if json:
            file_stream = file_io.FileIO(gcs_path, mode='rb')
            data = pd.read_json(BytesIO(file_stream.read()), encoding='utf-8',
                                compression='gzip', orient='records', lines=True)

        else:
            file_stream = file_io.FileIO(gcs_path, mode='rb')
            data = pd.read_csv(BytesIO(file_stream.read()), encoding='utf-8',
                               usecols=list_of_all_fields,
                               compression='gzip', header=header_rows, low_memory=False)
    else:

        if json:
            file_stream = file_io.FileIO(gcs_path, mode='r')
            data = pd.read_json(StringIO(file_stream.read()), encoding='utf-8',
                                compression=None, orient='records', lines=True)

        else:
            file_stream = file_io.FileIO(gcs_path, mode='r')
            data = pd.read_csv(StringIO(file_stream.read()), encoding='utf-8',
                               usecols=list_of_all_fields, compression=None, header=header_rows, low_memory=False)

    return data


def query_cached(bq_client, sql, save_file=None, overwrite=False):
    if not save_file:
        return bq_client.query(sql).to_dataframe()

    if not overwrite:
        try:
            df = pd.read_pickle(save_file)
            return df
        except Exception as e:
            print(e)

    df = bq_client.query(sql).to_dataframe()
    df.to_pickle(save_file)

    return df


def setup_logging_local(log_file_name, verbose=False):
    # get TF logger
    log = logging.getLogger('tensorflow')
    if verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    logFormatter = logging.Formatter(
        "%(asctime)s [%(filename)-20.20s:%(lineno)-4.4s - %(funcName)-20.20s() [%(threadName)-12.12s] [%(levelname)-8.8s]  %(message).5000s")

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file_name, encoding='utf8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logFormatter)
    log.addHandler(fh)

    # Set tensorflow verbosity
    if verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
        tf.logging.set_verbosity(tf.logging.DEBUG)
        tf.logging.debug("Set log level to debug.")
