import os

# How many CPU cores to use
CONCURRENCY = 31

TPU_ADDRESS = ""
FP16_IMPLEMENTATION = None
STRATEGY_TYPE = 'mirrored'
SCALE_LOSS = True
ENABLE_XLA = False  # Enables XLA in Session Config. Should not be set for TPU.
RUN_EAGERLY = False


# Project level defaults
TASK = 'twitter_abuse'  # @param {type:"string"}
TASK_DATA_DIR = 'classify_data/' + TASK
# Available pretrained model checkpoints:
#   uncased_L-12_H-768_A-12: uncased BERT base model
#   uncased_L-24_H-1024_A-16: uncased BERT large model
#   cased_L-12_H-768_A-12: cased BERT large model
BERT_MODEL = 'uncased_L-24_H-1024_A-16'  #@param {type:"string"}
BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL

PATH_TO_GOOGLE_KEY = '/Users/nic/src/platformgovernance/nicanalysis-c0df8a860fb.json'

TEXT_FIELDS = ['text']
LABEL_FIELD = 'classification'
ID_FIELD = 'id'

# CLASSIFICATION_CATEGORIES = ['nonabusive', 'abusive']
# We're trying out a new classification scheme
CLASSIFICATION_CATEGORIES = ['prohibited', 'misogynistic harassment', 'not relevant']

ALL_FIELDS = TEXT_FIELDS + [ID_FIELD] + [LABEL_FIELD]

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 1024
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 10.0
WARMUP_PROPORTION = 0.1
MAX_SEQUENCE_LENGTH = 64
SAVE_CHECKPOINTS_STEPS = 5000
SAVE_SUMMARY_STEPS = 100
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8

BUCKET = 'platform_governance_analysis'  #@param {type:"string"}
OUTPUT_DIR = 'gs://{}/{}/bert_models/'.format(BUCKET, TASK)

TRAINING_SETS = 'gs://{}/{}/training/*.csv'.format(BUCKET, TASK)  #@param {type:"string"}

VOCAB_FILE = 'gs://platform_governance_analysis/bert/BERT_uncased_L-24_H-1024_A-16_vocab-NS.txt'

PREDICT_DIR = 'gs://{}/{}/predictions/{}_{}'.format(BUCKET, TASK, BERT_MODEL, MAX_SEQUENCE_LENGTH)

PREDICT_TFRECORDS = 'gs://platform_governance_datasets/tfrecords_bert/twitter_tested_20181221_with_replacements/'
PREDICT_INPUT_PATH = 'gs://platform_governance_datasets/twitter_tested_20181221/twitter_tested_20181221*.csv.gz'

DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = 'gs://platform_governance_analysis/twitter_abuse/bert_models/model.ckpt-4227'

TRAIN_TFRECORDS = OUTPUT_DIR + 'train.tfrecords'
TEST_TFRECORDS = OUTPUT_DIR + 'test.tfrecords'
VALIDATION_TFRECORDS = OUTPUT_DIR + 'validation.tfrecords'

PREDICT_INPUT_GZIP = (PREDICT_INPUT_PATH[-2:] == 'gz')
PREDICT_INPUT_JSON = (PREDICT_INPUT_PATH[-4:] == 'json' or PREDICT_INPUT_PATH[-7:] == 'json.gz')
