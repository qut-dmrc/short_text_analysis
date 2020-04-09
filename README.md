# Short Text Analysis
A standard set of scripts for computational analysis of short text documents (particularly social media posts).

This includes some reusable classes and modules:

*  bert_train.py provides an interface to fine tune Google's pre-trained BERT model on sentence classification tasks.
*  bert_classify_tfrc.py uses a pre-trained model to runs predictions over a large dataset using Google's TensorFlow Research Cloud
*  processor_tfrecords.py efficiently turns dataframes into tensorflow records for deep learning
*  cloud_utils.py, plot_utils.py provide some general purpose functions.
