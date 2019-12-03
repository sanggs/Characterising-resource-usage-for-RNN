import collections

import helper
import argparse
import numpy as np
#import project_tests as tests
import tensorflow as tf
#from tensorflow.python.client import timeline
import GPUtil
import contextlib

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


from tensorflow.python.client import device_lib

class GPUUtilPrintingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.record = False

    def on_train_batch_end(self, batch, logs=None):
        if(self.record == True):
            if(batch == 2 or batch == 20):
                with open('logs/BRNN_GPU_Utils.txt', 'a') as f:
                    with contextlib.redirect_stdout(f):
                        print('Batch {} End.'.format(batch))
                        GPUtil.showUtilization()

    def on_train_batch_begin(self, batch, logs=None):
        if(self.record == True):
            if(batch == 2 or batch == 20):
                with open('logs/BRNN_GPU_Utils.txt', 'a') as f:
                    with contextlib.redirect_stdout(f):
                        print('Batch {} Begin.'.format(batch))
                        GPUtil.showUtilization()

    def on_epoch_begin(self, epoch, logs=None):
        if(epoch == 5):
            self.record = True
            with open('logs/BRNN_GPU_Utils.txt', 'a') as f:
                with contextlib.redirect_stdout(f):
                    print('Epoch {} Begin.'.format(epoch))
                    GPUtil.showUtilization()

    def on_epoch_end(self, epoch, logs=None):
        if(self.record == True):
            self.record = False
            with open('logs/BRNN_GPU_Utils.txt', 'a') as f:
                with contextlib.redirect_stdout(f):
                    print('Epoch {} End.'.format(epoch))
                    GPUtil.showUtilization()
                    print('---------------')


class Bidirectional:
    '''
    parser.add_argument("--epochs",type=int,default=2,help="Number of iterations to run the algorithm")
    parser.add_argument("--batch_size",type=int,default=1024,help="Batch size to consider for one gradient update")
    parser.add_argument("--learning_rate",type=float,default=0.01,help="Learning rate value")
    parser.add_argument("--validation_split",type=float,default=0.2,help="Validation fraction")
    parser.add_argument("--monitor",type=str,default='val_acc',help="Value to monitor for early stopping")
    parser.add_argument("--min_delta",type=float,default=1.0,help="Minimum increase/decrease in the monitored value")
    parser.add_argument("--patience",type=int,default=5,help="Minimum number of epochs to wati before triggering early stopping")
    '''
    def __init__(self, epochs=500, batch_size=512, learning_rate=1e-3, validation_split=0.2, monitor='val_acc', min_delta=0.0001, patience=50):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.sourceLanguage = None
        self.targetLanguage = None
        self.preprocessedSource = None
        self.preprocessedTarget = None
        self.sourceTokenizer = None
        self.targetTokenizer = None

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def setLearningRate(self, learning_rate):
        self.learning_rate = learning_rate

    def print_device(self):
        print(device_lib.list_local_devices())

    def loadData(self):
        # Load English data
        self.sourceLanguage = helper.load_data('data/small_vocab_en')
        # Load French data
        self.targetLanguage = helper.load_data('data/small_vocab_fr')
        print('Dataset Loaded')

    def seeSampleData(self):
        english_words_counter = collections.Counter([word for sentence in self.sourceLanguage for word in sentence.split()])
        french_words_counter = collections.Counter([word for sentence in self.targetLanguage for word in sentence.split()])

        print('{} English words.'.format(len([word for sentence in self.sourceLanguage for word in sentence.split()])))
        print('{} unique English words.'.format(len(english_words_counter)))
        print('10 Most common words in the English dataset:')
        print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
        print()
        print('{} French words.'.format(len([word for sentence in self.targetLanguage for word in sentence.split()])))
        print('{} unique French words.'.format(len(french_words_counter)))
        print('10 Most common words in the French dataset:')
        print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')

    def tokenize(self, x):
        """
        Tokenize x
        :param x: List of sentences/strings to be tokenized
        :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
        """
        # TODO: Implement
        x_tk = Tokenizer(char_level = False)
        x_tk.fit_on_texts(x)
        return x_tk.texts_to_sequences(x), x_tk

    def pad(self, x, length=None):
        """
        Pad x
        :param x: List of sequences.
        :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
        :return: Padded numpy array of sequences
        """
        # TODO: Implement
        if length is None:
            length = max([len(sentence) for sentence in x])
        return pad_sequences(x, maxlen = length, padding = 'post')

    def preprocess(self, x, y):
        """
        Preprocess x and y
        :param x: Feature List of sentences
        :param y: Label List of sentences
        :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
        """
        preprocess_x, x_tk = self.tokenize(x)
        preprocess_y, y_tk = self.tokenize(y)

        preprocess_x = self.pad(preprocess_x)
        preprocess_y = self.pad(preprocess_y)

        # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
        preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

        return preprocess_x, preprocess_y, x_tk, y_tk

    def prepareData(self):
        self.preprocessedSource, self.preprocessedTarget, self.sourceTokenizer, self.targetTokenizer = self.preprocess(self.sourceLanguage, self.targetLanguage)

        max_english_sequence_length = self.preprocessedSource.shape[1]
        max_french_sequence_length = self.preprocessedTarget.shape[1]
        english_vocab_size = len(self.sourceTokenizer.word_index)
        french_vocab_size = len(self.targetTokenizer.word_index)

        print('Data Preprocessed')
        print("Max English sentence length:", max_english_sequence_length)
        print("Max French sentence length:", max_french_sequence_length)
        print("English vocabulary size:", english_vocab_size)
        print("French vocabulary size:", french_vocab_size)

    def logits_to_text(self, logits, tokenizer):
        """
        Turn logits from a neural network into text using the tokenizer
        :param logits: Logits from a neural network
        :param tokenizer: Keras Tokenizer fit on the labels
        :return: String that represents the text of the logits
        """
        index_to_words = {id: word for word, id in tokenizer.word_index.items()}
        index_to_words[0] = '<PAD>'

        return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

        #print('`logits_to_text` function loaded.')


    def bd_model(self, input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
   
        model = Sequential()
        model.add(Bidirectional(GRU(128, return_sequences = True, dropout = 0.1), 
                               input_shape = input_shape[1:]))
        model.add(TimeDistributed(Dense(french_vocab_size, activation = 'softmax')))
        return model


    def train_model(self):
        #self.print_device()
        #self.loadData()
        #self.seeSampleData()
        #self.prepareData()
        x = self.pad(self.preprocessedSource, self.preprocessedTarget.shape[1])
        x = x.reshape((-1, self.preprocessedTarget.shape[-2], 1))
        x = np.float32(x)

        es = EarlyStopping(monitor=self.monitor, mode='auto', verbose=1, patience=self.patience)
        cb_list = [es, GPUUtilPrintingCallback()]

        bidi_model = self.bd_model(x.shape,
            self.preprocessedTarget.shape[1],
            len(self.sourceTokenizer.word_index)+1,
            len(self.targetTokenizer.word_index)+1)

        bidi_model.compile(loss = sparse_categorical_crossentropy, 
                     optimizer = Adam(args.learning_rate), 
                     metrics = ['accuracy'])

        print("The total number of trainable parameters are: " + str(bidi_model.count_params()))
        print("Model summary: ")
        print(bidi_model.summary())
        fileName = 'logs/brnn/b_'+str(self.batch_size)+'_lr_'+str(int(self.learning_rate*1e5))+'.txt'
        with open(fileName, 'w') as f:
            with contextlib.redirect_stdout(f):
                bidi_model.fit(x, self.preprocessedTarget, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split, callbacks=cb_list)

        print("training done")
        print("-------------")
        #print(self.logits_to_text(encodeco_model.predict(x[:1])[0], self.targetTokenizer))

