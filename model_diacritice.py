import tensorflow as tf
import numpy as np
import os
import unidecode
import re
import string
from tensorflow import keras
import nltk
from gensim.models.wrappers import FastText as FastTextWrapper
import threading

dict_lock = threading.Lock()
dict_avg_words = {}

window_sentence = 15 
window_character = 6 # 2 * x + 1

character_embedding_size = 20
word_embedding_size = 300

epochs = 15
reset_iterators_every_epochs = 5
characters_cell_size = 64
sentence_cell_size = 300
neurons_dense_layer_after_merge = 512
classes = 4
batch_size = 256
limit_backtracking_characters = 10

CPUS = 16
buffer_size_shuffle = 10000
max_unicode_allowed = 770
replace_character = 255
padding_character = 0

model_embeddings = FastTextWrapper.load_fasttext_format("fastText/wiki.ro")

# train_files = "small_train/"
# valid_files = "small_valid/"
# test_files = "small_test/"
train_files = "corpus/train/"
test_files = "corpus/test/"
valid_files = "corpus/validation/"
fast_text_embeddings_file = "fastText/wiki.ro.vec"

maps_no_diac = {
	'ă': 'a',
	'â': 'a',
	'Â': 'A',
	'Ă': 'A',
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

def get_clean_token(original_token):
	s = ""
	for c in original_token:
		if c in maps_no_diac:
			s += maps_no_diac[c]
		else:
			s += c
	return s

def check_similarity(clean_tokens, original_tokens):

	n = len(clean_tokens)

	for i in range(n):
		if len(clean_tokens[i]) != len(original_tokens[i]):
			return False

		for j in range(len(clean_tokens[i])):
			clean_token = get_clean_token(original_tokens[i][j])
			if clean_token != clean_tokens[i][j]:
				print(clean_token, clean_tokens[i][j])
				return False
	return True
			
def get_label(i, clean_text_utf, original_text_utf):

	case = 2
	if clean_text_utf[i] == 'a':
		if original_text_utf[i] == 'ă':
			case = 0
		elif original_text_utf[i] == 'â':
			case = 1
		elif original_text_utf[i] == 'a':
			case = 2
	elif clean_text_utf[i] == 'i':
		if original_text_utf[i] == 'î':
			case = 1
		elif original_text_utf[i] == 'i':
			case = 2
	elif clean_text_utf[i] == 't':
		if original_text_utf[i] == 'ț':
			case = 3
		elif original_text_utf[i] == 't':
			case = 2
	elif clean_text_utf[i] == 's':
		if original_text_utf[i] == 'ș':
			case = 3
		elif original_text_utf[i] == 's':
			case = 2

	label = np.float32([0] * 4)
	label[case] = np.float32(1.0)
	return label

def bkt_all_words(index, clean_word, current_word, maps_char_to_possible_chars):

	if index == len(clean_word):
		if current_word in model_embeddings.wv.vocab:
			return [current_word]
		else:
			return []
	else:
		L = []
		c = clean_word[index]
		if c in maps_char_to_possible_chars:
			for ch in maps_char_to_possible_chars[c]:
				L += bkt_all_words(index + 1, clean_word, \
					current_word + ch, maps_char_to_possible_chars)
		else:
			L += bkt_all_words(index + 1, clean_word, \
				current_word + c, maps_char_to_possible_chars)
		return L

def get_avg_possible_word(clean_word):

	with dict_lock:
		if clean_word in dict_avg_words:
			return dict_avg_words[clean_word]
		
	maps_char_to_possible_chars= {'a': ['ă', 'â', 'a'], 'i': ['î', 'i'],\
					 's': ['ș', 's'], 't': ['ț', 't']}

	count_diacritics_chars = 0
	for c in clean_word:
		if c in maps_char_to_possible_chars:
			count_diacritics_chars += 1
	
	if count_diacritics_chars > limit_backtracking_characters:
		return np.float32(model_embeddings.wv[clean_word])

	all_words = bkt_all_words(0, clean_word, "", maps_char_to_possible_chars)
	dict_words = []

	for word in all_words:
		#dict_words.append(np.random.rand(word_embedding_size))
		dict_words.append(np.float32(model_embeddings.wv[word]))

	if len(dict_words) > 0:
		possible_words = len(dict_words)
		final_embedding = np.float32(np.array([0] * word_embedding_size))
		for w_emb in dict_words:
			final_embedding += np.array(w_emb)
		final_embedding /= possible_words
		return final_embedding
	else:
		try:
			return np.float32(model_embeddings.wv[clean_word]) 
			#return np.float32([0] * word_embedding_size)
		except:
			return np.float32([0] * word_embedding_size)

def get_embeddings_sentence(clean_tokens_sentence, index_token):
	#if index_token == -1:
	#	return np.float32(np.zeros((window_sentence, word_embedding_size)))
	embeddings_sentence = []

	for i in range(-window_sentence, window_sentence + 1):
		if index_token + i >= 0 and index_token + i < len(clean_tokens_sentence):
			token = clean_tokens_sentence[index_token + i]
			try:
				embeddings_sentence.append(np.float32(model_embeddings.wv[token]))
				#embeddings_sentence.append(np.float32([0] * word_embedding_size))
			except:
				embeddings_sentence.append(np.float32([0] * word_embedding_size))
		else:
			embeddings_sentence.append(np.float32([0] * word_embedding_size))
	return np.array(embeddings_sentence)
	#return np.array(embeddings_sentence)

# discard first chars that are not included in the tokenization
# in order to not associate wrong tokens for these chars
def discard_first_chars(index_text, clean_text_utf, clean_token):

	cnt_chars_token = 0
	char_token = clean_token[0]
	# discard chars which are not the same in text
	while index_text < len(clean_text_utf) and clean_text_utf[index_text] != char_token:
		index_text += 1
	# count chars which are the same in text
	while index_text < len(clean_text_utf) and clean_text_utf[index_text] == char_token:
		index_text += 1
	# count chars which are the same in token
	while cnt_chars_token < len(clean_token) and clean_token[cnt_chars_token] == char_token:
		cnt_chars_token += 1
	
	return index_text - cnt_chars_token

# return an tuple input (window_char, embedding_token, embedding_sentence)
def get_input_example(clean_text_utf, index_text, clean_tokens, index_sent, index_token):
	
	w = [] # window with characters
	# create window of characters
	for j in range(-window_character, window_character + 1):
		if index_text + j < 0 or index_text + j >= len(clean_text_utf):
			v1 = padding_character
		elif ord(clean_text_utf[index_text + j]) > max_unicode_allowed:
			v1 = replace_character
		else:
			v1 = ord(clean_text_utf[index_text + j])
		w.append(v1)

	token_embedding = get_avg_possible_word(clean_tokens[index_sent][index_token])
	with dict_lock:
		dict_avg_words[dict_lock] = token_embedding

	sentence_embedding = get_embeddings_sentence(clean_tokens[index_sent], index_token)
	return (np.int32(w), token_embedding, sentence_embedding)
	#return np.int32(np.array([0])), np.int32(np.array([0, 0])),  np.int32(np.array([0, 0, 0]))

def create_examples(clean_text, original_text):
	clean_text_utf = clean_text.decode('utf-8')
	# replace some strange characters which are modified by tokenization
	substitute = {'"': '`'}
	clean_text_utf_replaced = ""
	for c in clean_text_utf:
		if ord(c) > 255:
			clean_text_utf_replaced += chr(replace_character)
		else:
			clean_text_utf_replaced += substitute[c] if c in substitute else c

	clean_text_utf = clean_text_utf_replaced
	original_text_utf = original_text.decode('utf-8')
	
	clean_sentences = nltk.sent_tokenize(clean_text_utf)

	clean_tokens = []

	# construct tokens
	for i in range(len(clean_sentences)):
		clean_tokens_sent = nltk.word_tokenize(clean_sentences[i])
		clean_tokens.append(clean_tokens_sent)

	index_text = 0 # current position in text
	index_sent = 0 # current sentence
	index_token = 0 # current token
	window_characters = []
	word_embeddings = []
	sentence_embeddings = []
	labels = []
	while index_sent < len(clean_tokens):
		clean_token = clean_tokens[index_sent][index_token]
#		index_text = discard_first_chars(index_text, clean_text_utf, clean_token)
		i = 0		
		while i < len(clean_token):
			label = get_label(index_text, clean_text_utf, original_text_utf)
			win_char, word_emb, sent_emb = get_input_example(clean_text_utf, \
						index_text, clean_tokens, index_sent, index_token)
			if clean_text_utf[index_text] == clean_token[i]:
				index_text += 1
				i += 1
			else: # discard char in text
				index_text += 1
			window_characters.append(win_char)
			word_embeddings.append(word_emb)
			sentence_embeddings.append(sent_emb)
			labels.append(label)

		if index_token == len(clean_tokens[index_sent]) - 1:
			index_token = 0
			index_sent += 1
		else:
			index_token += 1

	return (window_characters, word_embeddings, sentence_embeddings, labels)

def filter_null_strings(s):
	if len(s) == 0:
		return np.array([False])
	return np.array([True])

def flat_map_f(a, b):
	return tf.data.Dataset.from_tensor_slices((a, b))

def get_dataset(dpath, sess):

	to_lower = create_lower_mapping()
	input_files = tf.gfile.ListDirectory(dpath)
	for i in range(len(input_files)):
		input_files[i] = dpath + input_files[i]

	# correct diac
	dataset = tf.data.TextLineDataset(input_files)
	dataset = dataset.filter(lambda x:
	 (tf.py_func(filter_null_strings, [x], tf.bool, stateful=False))[0])

	for m in to_lower:
		dataset = dataset.map(lambda x: 
			tf.regex_replace(x, tf.constant(m), tf.constant(to_lower[m])), num_parallel_calls=CPUS)
	dataset = dataset.map(lambda x: (x, x), num_parallel_calls=CPUS)

	for m in correct_diac:
		dataset = dataset.map(lambda x, y:
			(tf.regex_replace(x, tf.constant(m), tf.constant(correct_diac[m])),
			tf.regex_replace(y, tf.constant(m), tf.constant(correct_diac[m])),), num_parallel_calls=CPUS)
		
	for m in maps_no_diac:
	 	dataset = dataset.map(lambda x, y: 
			(tf.regex_replace(x, tf.constant(m), tf.constant(maps_no_diac[m])), y), num_parallel_calls=CPUS)

	dataset = dataset.map(lambda x, y: 
		tf.py_func(create_examples, (x, y), (tf.int32, tf.float32, tf.float32, tf.float32), stateful=False), num_parallel_calls=CPUS)

	dataset = dataset.map(lambda x1, x2, x3, y: ((x1, x2, x3), y), num_parallel_calls=CPUS)
	dataset = dataset.flat_map(flat_map_f)
	
	filter_chars = lambda x, y: \
		tf.logical_or( tf.logical_or(tf.equal(x[0][window_character + 1], ord('a')), \
								tf.equal(x[0][window_character + 1], ord('i'))), \
								tf.logical_or(tf.equal(x[0][window_character + 1], ord('t')), \
								tf.equal(x[0][window_character + 1], ord('s'))))
	
	dataset = dataset.filter(filter_chars)
	dataset = dataset.shuffle(buffer_size_shuffle)
	dataset = dataset.batch(batch_size)

	return dataset

with tf.Session() as sess:

	dt_train = get_dataset(train_files, sess)
	dt_valid = get_dataset(valid_files, sess)
	dt_test = get_dataset(test_files, sess)

	inp_batches_train = 380968863 // batch_size
	inp_batches_test = 131424533 // batch_size
	inp_batches_valid = 131861863 // batch_size

	# inp_batches_train = 10650 // batch_size
	# inp_batches_test = 3978 // batch_size
	# inp_batches_valid = 50490 // batch_size

	vocabulary_size = max_unicode_allowed + 1

	iterator_train = dt_train.make_initializable_iterator()
	iterator_valid = dt_valid.make_initializable_iterator()
	iterator_test = dt_test.make_initializable_iterator()

	sess.run(iterator_test.initializer)

	# character window 
	input_character_window = keras.layers.Input(shape=(window_character * 2 + 1,))
	character_embeddings_layer = keras.layers.Embedding(\
								input_dim=vocabulary_size,\
								output_dim=character_embedding_size)(input_character_window)

	character_lstm_layer = keras.layers.LSTM(
							units=characters_cell_size,\
							input_shape=(window_character * 2 + 1, character_embedding_size,))

	characters_bi_lstm_layer = keras.layers.Bidirectional(
							layer=character_lstm_layer,\
							merge_mode="concat")(character_embeddings_layer)
	# word token					
	word_embeddings_layer = keras.layers.Input(shape=(word_embedding_size,))

	# sentence token
	sentence_embeddings_layer = keras.layers.Input(shape=((window_sentence * 2 + 1, word_embedding_size,)))
	sentence_lstm_layer = keras.layers.LSTM(units=sentence_cell_size,\
											input_shape=(window_sentence * 2 + 1, word_embedding_size,))	

	sentence_bi_lstm_layer = keras.layers.Bidirectional(layer=sentence_lstm_layer,\
                                                        merge_mode="concat")(sentence_embeddings_layer)
	merged_layer = keras.layers.concatenate([characters_bi_lstm_layer, \
				word_embeddings_layer, sentence_bi_lstm_layer], axis=-1)

	dense_layer = keras.layers.Dense(neurons_dense_layer_after_merge, activation='tanh')(merged_layer)
	output = keras.layers.Dense(classes, activation='softmax')(dense_layer)

	model = keras.models.Model(inputs=[input_character_window, word_embeddings_layer, sentence_embeddings_layer],\
	 						   outputs=output)
	model.compile(optimizer='adam',\
				  loss='categorical_crossentropy',\
				  metrics=['accuracy', keras.metrics.categorical_accuracy])
	#train_inp, train_out = iterator_train.get_next()
	#valid_inp, valid_out = iterator_valid.get_next()
	test_inp, test_out = iterator_test.get_next()

	#train_char_window, train_words, train_sentence = train_inp
	#valid_char_window, valid_words, valid_sentence = valid_inp
	test_char_window, test_words, test_sentence = test_inp

	for i in range(epochs):
		if i % reset_iterators_every_epochs == 0:
			sess.run(iterator_valid.initializer)
			valid_inp, valid_out = iterator_valid.get_next()
			valid_char_window, valid_words, valid_sentence = valid_inp

			sess.run(iterator_train.initializer)
			train_inp, train_out = iterator_train.get_next()
			train_char_window, train_words, train_sentence = train_inp
			
		model.fit(\
			 [train_char_window, train_words, train_sentence],\
			 [train_out],\
			 steps_per_epoch=inp_batches_train//reset_iterators_every_epochs,
			 epochs=1, 
			 verbose=1)
		
		[valid_loss, valid_acc, valid_cross_entropy] = model.evaluate([valid_char_window, valid_words, valid_sentence],\
									valid_out,\
									verbose=1,\
									steps=inp_batches_valid//reset_iterators_every_epochs)
		print("validation - loss: " + str(valid_loss) +  " acc: " + str(valid_acc))

	# [test_loss, test_acc, test_cross_entropy] = model.evaluate(\
	# 								[test_char_window, test_words, test_sentence],\
	# 								test_out,\
	# 								verbose=1,\
	# 								steps=inp_batches_test)
	# print("test - loss: " + str(test_loss) +  " acc: " + str(test_acc))
