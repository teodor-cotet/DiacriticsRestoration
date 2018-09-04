import tensorflow as tf
import numpy as np
import os
import unidecode
import re
import string
import tensorflow.keras as keras

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
window_size = 12
epochs = 5
embedding_size = 20
cell_size = 64
classes = 4
buffer_size_shuffle = 100000
max_unicode_allowed = 770
safety_batches = 10000
batch_size = 256

train_files = "corpus/train/"
test_files = "corpus/test/"
valid_files = "corpus/validation/"

maps_no_diac = {
	'ă|â': 'a',
	'Ă|Â': 'A',
	'ț': 't',
	'Ț': 'T',
	'ș': 's',
	'Ș': 'S',
	'î': 'i',
	'Î': 'I'
}

correct_diac = {
	"ş": "ș",
	"Ş": "Ș",
	"ţ": "ț",
	"Ţ": "Ț",
}

def create_lower_mapping():
	to_lower = { }
	for c in  string.ascii_uppercase:
		to_lower[c] = c.lower()
	to_lower['Ș'] = 'ș'
	to_lower['Â'] = 'â'
	to_lower['Ă'] = 'ă'
	to_lower['Ț'] = 'ț'
	to_lower['Î'] = 'î'
	return to_lower

def create_windows(s, s2):
	su = s.decode('utf-8')
	s2u = s2.decode('utf-8')
	windows = []
	labels = []

	for i in range(len(su)):
		w = []
		for j in range(-window_size, window_size + 1):
			if i + j < 0 or i + j >= len(su):
				v1 = np.int32(0)
			elif ord(su[i + j]) > max_unicode_allowed:
				v1 = np.int32(767)
			else:
				v1 = np.int32(ord(su[i + j]))

			w.append(v1)
		windows.append(w)
		case = 2
		# labels.append(np.float32(ord(s2u[i])))
		if su[i] == 'a':
			if s2u[i] == 'ă':
				case = 0
			elif s2u[i] == 'â':
				case = 1
			elif s2u[i] == 'a':
				case = 2
		elif su[i] == 'i':
			if s2u[i] == 'î':
				case = 1
			elif s2u[i] == 'i':
				case = 2
		elif su[i] == 't':
			if s2u[i] == 'ț':
				case = 3
			elif s2u[i] == 't':
				case = 2
		elif su[i] == 's':
			if s2u[i] == 'ș':
				case = 3
			elif s2u[i] == 's':
				case = 2

		l = np.float32([0] * 4)
		l[case] = np.float32(1.0)
		labels.append(l)

	return (windows, labels)

def filter_func(s):
	if len(s) == 0:
		return np.array([False])
	return np.array([True])

def flat_map_f(a, b):
	return tf.data.Dataset.from_tensor_slices((a, b))

def filt(x):
	if x[window_size + 1] == ord('a') or \
	x[window_size + 1] == ord('t') or \
	x[window_size + 1] == ord('i') or \
	x[window_size + 1] == ord('s'):
		return np.array([True])
	return np.array([False])

def get_dataset(dpath):

	to_lower = create_lower_mapping()
	input_files = tf.gfile.ListDirectory(dpath)
	for i in range(len(input_files)):
		input_files[i] = dpath + input_files[i]

	# correct diac
	dataset = tf.data.TextLineDataset(input_files)
	dataset = dataset.filter(lambda x:
	 (tf.py_func(filter_func, [x], tf.bool, stateful=False))[0])
	
	for m in to_lower:
		dataset = dataset.map(lambda x: 
			tf.regex_replace(x, tf.constant(m), tf.constant(to_lower[m])))

	dataset = dataset.map(lambda x: (x, x))

	for m in correct_diac:
		dataset = dataset.map(lambda x, y:
			(tf.regex_replace(x, tf.constant(m), tf.constant(correct_diac[m])),
			tf.regex_replace(y, tf.constant(m), tf.constant(correct_diac[m]))))
		
	for m in maps_no_diac:
	 	dataset = dataset.map(lambda x, y: 
			(tf.regex_replace(x, tf.constant(m), tf.constant(maps_no_diac[m])), y))

	dataset = dataset.map(lambda x, y: 
		tf.py_func(create_windows, (x, y), (tf.int32, tf.float32), stateful=False))

	dataset = dataset.flat_map(flat_map_f)
	dataset = dataset.filter(lambda x, y:
	 (tf.py_func(filt, [x], tf.bool, stateful=False))[0])
	dataset = dataset.shuffle(buffer_size_shuffle)
	
	dataset = dataset.batch(batch_size)

	iterator = dataset.make_initializable_iterator()
	sess.run(iterator.initializer)
	next_element = iterator.get_next()
	return dataset

with tf.Session() as sess:

	dt_train = get_dataset(train_files)
	dt_valid = get_dataset(valid_files)
	# dt_test = get_dataset(test_files)

	inp_batches_train = 380968863 // batch_size
	inp_batches_test = 131424533 // batch_size
	inp_batches_valid = 131861863 // batch_size

	vocabulary_size = max_unicode_allowed + 1

	iterator_train = dt_train.make_initializable_iterator()
	# iterator_test = dt_test.make_initializable_iterator()
	iterator_valid = dt_valid.make_initializable_iterator()

	sess.run(iterator_train.initializer)
	#sess.run(iterator_test.initializer)
	sess.run(iterator_valid.initializer)

	model = keras.models.Sequential()
	model.add(keras.layers.Embedding(vocabulary_size, 
		embedding_size, input_length=window_size * 2 + 1))
	model.add(keras.layers.Bidirectional(keras.layers.LSTM(cell_size)))
	model.add(keras.layers.Dense(64, activation = 'tanh'))
	model.add(keras.layers.Dense(classes, activation = 'softmax'))

	model.compile(optimizer='adam', loss='categorical_crossentropy', 
		metrics = ['acc'])
	
	for _ in range(epochs):
		model.fit(iterator_train, 
			 steps_per_epoch=inp_batches_train,
			 epochs=1, 
			 verbose=1,
			 validation_data=iterator_valid,
			 validation_steps=inp_batches_valid)
