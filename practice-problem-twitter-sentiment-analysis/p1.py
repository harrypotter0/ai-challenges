# -*- coding: utf-8 -*-
"""
@author: Akash Kandpal
@date: 24 May 2018
Analytics Vidhya Datahack : Twitter Sentiment Analysis
Algorithm: Convolutional Neural Network (CNN) with Word Embedding
"""

import pandas
import re
import string
import tensorflow as tf
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

#---------------------------------
# BUILT-IN FUNCTIONS
#---------------------------------
def clean_text(txt):
    """Preprocessing - Turning texts into clean tokens
    """
    # Ensure lowercase text encoding
    txt = str(txt).lower()
    # split tokens by white space
    tokens = txt.split()
    # remove tokens not encoded in ascii
    isascii = lambda s: len(s) == len(s.encode())
    tokens = [w for w in tokens if isascii(w)]
    # regex for punctuation filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove tokens that aren't alphanumeric
    tokens = [w for w in tokens if w.isalnum()]
    # regex for digits filtering
    re_digt = re.compile('[%s]' % re.escape(string.digits))
    # remove digits from each word
    tokens = [re_digt.sub('', w) for w in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out long tokens
    tokens = [w for w in tokens if len(w) < 30]
    # filter out short tokens
    tokens = [w for w in tokens if len(w) > 1]
    # stemming of words
    porter = PorterStemmer()
    tokens = [porter.stem(w) for w in tokens]
    return tokens

def token_to_line(txt, vocab):
    """Clean text and return line of tokens
    dependency: clean_text
    """
    # clean text
    tokens = clean_text(txt)
    # filter by vocabulary
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def process_texts(texts, vocab):
    """Clean texts to only contain tokens present in the vocab
    dependency: token_to_line
    """
    lines = list()
    for txt in texts:
        # load and clean the doc
        line = token_to_line(txt, vocab)
        # add to list
        lines.append(line)
    return lines

def save_vocab(lines, filename):
    """Saving a list of items to a file; line-by-line
    """
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def load_vocab(filename):
    """Load doc into memory
    """
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def add_tokens_vocab(txt, vocab):
    """Creating vocabulary containing unique tokens from all texts
    dependency: add_tokens_vocab
    """
    tokens = clean_text(txt)
    vocab.update(tokens)

def build_vocab(texts):
    """Creating vocabulary and saving output to a text file
    dependency: clean_text
    """
    vocab = Counter()
    for txt in texts:
        add_tokens_vocab(txt, vocab)
    # save tokens to a vocabulary file; for later access in model build/predict
    save_vocab(vocab, data_path + "/vocab.txt")

def create_tokenizer(lines):
    """ Defining a tokenizer
    dependency: from keras.preprocessing.text import Tokenizer
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def encode_docs(tokenizer, max_length, docs):
    """ Encode each 'cleaned' string as a sequence of integers
    dependency: create_tokenizer
    """
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences to ensure that all strings have the same length
    # max_length is the length of the longest string
    padded = pad_sequences(encoded, maxlen = max_length, padding='post')
    return padded

def tf_auc_roc(y_true, y_pred):
    """ Defining AUC ROC metrics for model performance from tensorflow package since AUC isn't available in Keras
    dependency: import tensorflow as tf
    """
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
    # Add metric variables to GLOBAL_VARIABLES collection.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def define_model(vocab_size, max_length):
    """ Defining the neural network model
    """
    model = Sequential()
    # embedding part: 150-dimensional vector space (explicit assignment; experimental)
    model.add(Embedding(vocab_size, 150, input_length = max_length))
    # add a CNN layer with 32 filters (parallel fields for processing words)
    #   and a kernel size of 8 with a rectified linear (relu) activation function.
    model.add(Conv1D(filters = 32, kernel_size = 8, activation='relu'))
    # add pooling layer to reduce the output of the CNN layer
    #   pool_size = 2 to reduce by half
    model.add(MaxPooling1D(pool_size=2))
    # flatten the CNN output to one long 2D vector representing features extracted by CNN
    model.add(Flatten())
    # add a standard MLP layer to interpret the CNN features
    model.add(Dense(30, activation='relu'))
    # use a sigmoid activation function in the output layer to resturn a value between 0 and 1 (binary classification)
    model.add(Dense(1, activation='sigmoid'))
    return model


#---------------------------------
# MAIN
#---------------------------------
# data_path = "/Users/bryanbalajadia/DataScience/GitHub_repos/AVDatahack_TwitterSentimentAnalysis/Data"

print("Loading data sets into Memory...")
train_df = pandas.read_csv("train.csv", quotechar='"', skipinitialspace=True, encoding='utf-8')
print("...training data dimension: " + str(train_df.shape))
test_df = pandas.read_csv("test.csv", quotechar='"', skipinitialspace=True, encoding='utf-8')
print("...test data (rows for prediction): " + str(test_df.shape[0]))

#----
# Data prep to Model Build
#----

print("Shuffling the training data row-wise...")
# Shuffle the data frame row-wise
#   useful during model fit since keras is getting only the last n% of data (w/o randomization)
#       in defining the validation set
train_df = train_df.sample(frac=1).reset_index(drop=True)

print("Building the neural network inputs...")
# get target
ytrain = train_df.label

# create vocabulary file from the train data
build_vocab(train_df.tweet)

# load the vocabulary
tokens = load_vocab(data_path + "/vocab.txt")

# # process strings to contain only clean tokens
# texts = process_texts(train_df.tweet, vocab = tokens)
#
# # identify the maximum string word length
# max_length = max([len(t.split()) for t in texts])
#
# # instantiate the tokenizer
# tokenizer = create_tokenizer(texts)
#
# # identify the size of the full vocabulary
# #   add +1 for unknown words
# vocab_size = len(tokenizer.word_index) + 1
#
# # prepare train and test sets for network processing
# xtrain = encode_docs(tokenizer, max_length, texts)
# xtest = encode_docs(tokenizer, max_length, process_texts(test_df.tweet, vocab = tokens))
#
# print("Defining the neural network model...")
# # define the neural network model
# model = define_model(vocab_size, max_length)
#
# # compile network
# #   use binary cross entropy loss function for classification problem
# #   use the 'adam' implementation of stochastic gradient descent
# #   keep track of AUC ROC in addition to loss during training
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf_auc_roc])
#
# # summarize the defined network
# model.summary()
#
# # model checkpoint
# #   checkpointing to ensure that each time the model performance improves on the validation set during model build,
# #   the model is saved to file.
# #   performance is evaluated based on the defined AUC ROC (monitor='val_tf_auc_roc')
# filepath = data_path + "/weights.bestmodel.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_tf_auc_roc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
#
# print("Running model Build...")
# # fit network
# #   30 epochs to cycle through the training data (can be configured differently)
# #   set the last 10% of training data as the validation set (can be configured differently)
# #   use pre-defined callback_list to save the best model in the cycle
# #   Assign class_weight to handle data imbalance
# class_weight = {0 : 1000., 1: 75.}
# model.fit(xtrain, ytrain, epochs = 30, validation_split = 0.10, verbose = 0, callbacks=callbacks_list, class_weight = class_weight)
# print("...Model build process:COMPLETED")
#
# #----
# # Test file prediction to writing a .csv submission file
# #----
#
# # redefine the network structure (can be skipped if model build is active in the current session)
# model = define_model(vocab_size, max_length)
# model.load_weights(data_path + "/weights.bestmodel.hdf5")
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf_auc_roc])
#
# print("Running test set predictions...")
# # run predictions
# results = pandas.DataFrame(model.predict(xtest, verbose=0))
#
# print("writing predictions to a submission file...")
# # write a submission file
# test_df["label"] = results.iloc[:,0]
# test_df["label"] = round(test_df["label"])
# test_df = test_df[["id", "label"]]
# test_df.to_csv(data_path + "/test_predictions.csv", encoding='utf-8',index=False)
# print("...Test set prediction process:COMPLETED")
#
#
# #---End-Of-File
