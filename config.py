import os

import tensorflow as tf

CONCURRENCY = 7

# Project level defaults
TASK = 'task_name'
TASK_DATA_DIR = 'classify_data/' + TASK

# Google JSON service account key for accessing GCS and TPUs
PATH_TO_GOOGLE_KEY = '/Users/nic/src/platformgovernance/nicanalysis-c0df8a860fb.json'

# Available pretrained model checkpoints:
#   uncased_L-12_H-768_A-12: uncased BERT base model
#   uncased_L-24_H-1024_A-16: uncased BERT large model
#   cased_L-12_H-768_A-12: cased BERT large model
BERT_MODEL = 'uncased_L-24_H-1024_A-16'
BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
tf.logging.info('***** Task data directory: {} *****'.format(TASK_DATA_DIR))
tf.logging.info('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

BUCKET = ''  # Add your GCS bucket for output
assert BUCKET, 'Must specify an existing GCS bucket name'
OUTPUT_DIR = 'gs://{}/{}/bert_models/'.format(BUCKET, TASK)
tf.logging.info('***** Model output directory: {} *****'.format(OUTPUT_DIR))

TRAINING_SETS = 'gs://{}/{}/training/*.csv'.format(BUCKET, TASK)

PREDICT_DIR = 'gs://{}/{}/predictions/'.format(BUCKET, TASK)
tf.logging.info('***** Predictions directory: {} *****'.format(PREDICT_DIR))

VOCAB_FILE = 'gs://platform_governance_analysis/bert/BERT_uncased_L-24_H-1024_A-16_vocab-NS.txt'
tf.logging.info('***** Vocabulary file: {} *****'.format(VOCAB_FILE))

DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

TEXT_FIELDS = []  # A list of all the text fields you want to operate on
LABEL_FIELD = 'classification'  # The name of the field with the label or classification
ID_FIELD = ''  # A field with a unique identifier for each record
CLASSIFICATION_CATEGORIES = []  # Your list of categories
ALL_FIELDS = TEXT_FIELDS + [ID_FIELD] + [LABEL_FIELD]

# BERT Model Hyper Parameters
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 10.0
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 64
MAX_SEQUENCE_LENGTH = MAX_SEQ_LENGTH

SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8

USE_MULTI_GPU = False

CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
#INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'model.ckpt')

# PREDICT_INPUT_PATH should be a path specification for CSV or JSON records
# that we will transform to TFRecords
PREDICT_INPUT_PATH = 'gs://platform_governance_datasets/twitter_tested_20181221/*.csv.gz'
# GCS_INPUT_PATH = 'gs://platform_governance_datasets/youtube_tested_20190204/*.json.gz'  # @param {type:"string"}
PREDICT_INPUT_GZIP = (PREDICT_INPUT_PATH[-2:] == 'gz')
PREDICT_INPUT_JSON = (PREDICT_INPUT_PATH[-4:] == 'json' or PREDICT_INPUT_PATH[-7:] == 'json.gz')

PREDICT_TFRECORDS = ''  # GCS path with TFRecords to predict
PREDICT_TFRECORDS = 'gs://platform_governance_datasets/tfrecords_bert/twitter_tested_20181221_with_replacements/'  # @param {type:"string"}
assert PREDICT_TFRECORDS, 'Must specify an existing GCS bucket name'


# Available pretrained model checkpoints:
#   uncased_L-12_H-768_A-12: uncased BERT base model
#   uncased_L-24_H-1024_A-16: uncased BERT large model
#   cased_L-12_H-768_A-12: cased BERT large model

# Custom dictionary - based off the original BERT dictionary, but with additional special tokens replacing unused tokens
# '/db/pretrained/bert/uncased_L-24_H-1024_A-16/vocab.txt' #os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')

if LABEL_FIELD:
    ALL_FIELDS.append(LABEL_FIELD)
