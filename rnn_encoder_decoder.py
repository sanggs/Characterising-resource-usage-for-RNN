import collections

import helper
import argparse
import numpy as np
#import project_tests as tests
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
print(device_lib.list_local_devices())

# Load English data
english_sentences = helper.load_data('data/small_vocab_en')
# Load French data
french_sentences = helper.load_data('data/small_vocab_fr')

print('Dataset Loaded')

english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    x_tk = Tokenizer(char_level = False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))


def pad(x, length=None):
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

#tests.test_pad(pad)

# Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))




def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('`logits_to_text` function loaded.')


def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  
    learning_rate = 1e-3
    model = Sequential()
    model.add(GRU(128, input_shape = input_shape[1:], return_sequences = False))
    model.add(RepeatVector(output_sequence_length))
    model.add(GRU(128, return_sequences = True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation = 'softmax')))
    
    # model.compile(loss = sparse_categorical_crossentropy, 
    #              optimizer = Adam(learning_rate), 
    #              metrics = ['accuracy'])

    return model

def main(args):
    x = pad(preproc_english_sentences)
    x = x.reshape((-1, preproc_english_sentences.shape[1], 1))
    x = np.float32(x)

    es = EarlyStopping(monitor=args.monitor, mode='auto', verbose=1, patience=args.patience)
    cb_list = [es]

    encodeco_model = encdec_model(x.shape, preproc_french_sentences.shape[1],
        len(english_tokenizer.word_index)+1,len(french_tokenizer.word_index)+1)

    encodeco_model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(args.learning_rate), 
                 metrics = ['accuracy'])

    print("The total number of trainable parameters are: " + str(encodeco_model.count_params()))
    print("Model summary: ")
    print(encodeco_model.summary())

    encodeco_model.fit(x, preproc_french_sentences, epochs=args.epochs, batch_size=args.batch_size, 
                        validation_split=args.validation_split, callbacks=cb_list)

    print(logits_to_text(encodeco_model.predict(x[:1])[0], french_tokenizer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=2,help="Number of iterations to run the algorithm")
    parser.add_argument("--batch_size",type=int,default=1024,help="Batch size to consider for one gradient update")
    parser.add_argument("--learning_rate",type=float,default=0.01,help="Learning rate value")
    parser.add_argument("--validation_split",type=float,default=0.2,help="Validation fraction")
    parser.add_argument("--monitor",type=str,default='val_accuracy',help="Value to monitor for early stopping")
    parser.add_argument("--min_delta",type=float,default=1.0,help="Minimum increase/decrease in the monitored value")
    parser.add_argument("--patience",type=int,default=5,help="Minimum number of epochs to wati before triggering early stopping")
    args, unknown = parser.parse_known_args()
    main(args)

#tests.test_encdec_model(encdec_model)
tmp_x = pad(preproc_english_sentences)
tmp_x = tmp_x.reshape((-1, preproc_english_sentences.shape[1], 1))
tmp_x = np.float32(tmp_x)

encodeco_model = encdec_model(
    tmp_x.shape,
    preproc_french_sentences.shape[1],
    len(english_tokenizer.word_index)+1,
    len(french_tokenizer.word_index)+1)
encodeco_model.fit(tmp_x, preproc_french_sentences, batch_size=128, epochs=10, validation_split=0.2)
print(logits_to_text(encodeco_model.predict(tmp_x[:1])[0], french_tokenizer))




