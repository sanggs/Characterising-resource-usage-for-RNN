import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_data(path):
	input_file = os.path.join(path)
	with open(input_file, "r", encoding="utf-8") as f:
		data = f.read()

	return data.split('\n')


def tokenize(x):
	x_tk = Tokenizer(char_level = False)
	x_tk.fit_on_texts(x)
	return x_tk.texts_to_sequences(x), x_tk


def pad(x, length=None):
	if length is None:
		length = max([len(sentence) for sentence in x])
	return pad_sequences(x, maxlen = length, padding = 'post')

def index_to_words(logits, tokenizer):
	i2w = {id: word for word, id in tokenizer.word_index.items()}
	i2w[0] = '<PAD>'
	return ' '.join([i2w[prediction] for prediction in np.argmax(logits, 1)])

def getDataFrame(train_data, test_data):
	train_seq = tokenize(train_data)
	test_seq = tokenize(test_data)
	x_train = pd.DataFrame.from_dict({
		'trainData': train_data,
		'iTrainData': train_seq
	})

	x_test = pd.DataFrame.from_dict({
		'labelData': test_data,
		'iLabelData': test_seq
	})

	x_train['lengthTrain'] = x_train.iTrainData.str.len()
	x_train['iLabelData'] = x_test.iLabelData
	x_train['labelData'] = x_test.labelData
	x_train['lengthTest'] = x_test.iLabelData.str.len()
	return x_train
