import collections
import datetime
import itertools
import math
import multiprocessing as mp
import re

import pandas as pd
import tensorflow as tf
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype

from cloud_utils import read_df_gcs
from tensorflow_models.official.nlp.bert import tokenization, run_classifier


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
    for train_file in tf.io.gfile.glob(gcs_path):
        data = read_df_gcs(train_file, list_of_all_fields)
        tf.compat.v1.logging.info('Found columns: {}'.format(data.columns))
        tf.compat.v1.logging.info('Found categories: {}'.format(data[label_field].unique()))
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

    tf.compat.v1.logging.info("Found {} rows total.".format(df.shape[0]))

    return df


def preprocess_df(df, list_of_all_fields, list_of_text_fields, label_field, id_field):
    """ Pre-process the dataframe - wrangle the columns into a standard format."""
    tf.compat.v1.logging.info('Starting to preprocess dataframe containing {} rows.'.format(df.shape[0]))

    tf.compat.v1.logging.debug(f"Found columns: {df.columns}")
    tf.compat.v1.logging.debug(f"Expecting columns: {list_of_all_fields}")

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

    if label_field and is_string_dtype(df[label_field]):
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


def convert_df_to_examples_mp(df, concurrency, vocab_file, do_lower_case, label_list, max_seq_length,
                              is_predicting=False):
    batch_size = int(math.ceil(df.shape[0] / concurrency))
    list_df = [df[i:i + batch_size] for i in range(0, df.shape[0], batch_size)]
    tf.compat.v1.logging.info('Chunked dataframe into {} chunks.'.format(len(list_df)))

    with mp.Pool(processes=concurrency) as pool:
        # results = pool.map(convert_dataframe_to_examples, list_df)
        results = parmap.map(convert_dataframe_to_examples, list_df, vocab_file, do_lower_case, label_list,
                             max_seq_length, is_predicting)

    flattened_list_of_examples = list(itertools.chain(*results))
    tf.compat.v1.logging.info('Finished processing dataframe.')

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
    tf.compat.v1.logging.info("Finished converting dataframe chunk in {}.".format(time_taken))

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
    tf.compat.v1.logging.info("Saving {} to {} and {}.".format(len(list_of_examples), output_file_examples, output_file_guids))
    example_writer = tf.io.TFRecordWriter(output_file_examples)
    guid_writer = file_io.FileIO(output_file_guids, mode='w')

    for guid, example in list_of_examples:
        example_writer.write(example.SerializeToString())
        guid_writer.write(guid + "\n")
    guid_writer.close()
    example_writer.close()
    tf.compat.v1.logging.info("Done saving.")