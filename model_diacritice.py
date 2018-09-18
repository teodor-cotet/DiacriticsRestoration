import tensorflow as tf
import numpy as np
import os
import unidecode
import re
import string
from tensorflow import keras
from typing import List
import time

import nltk
from gensim.models.wrappers import FastText as FastTextWrapper
import threading

dict_lock = threading.Lock()
dict_avg_words = {}

total_time_tokenization = 0
total_time = 0
total_time_bkt = 0
total_time_sent = 0

window_sentence = 15 
window_character = 6 # 2 * x + 1

character_embedding_size = 20
word_embedding_size = 300

epochs = 20
reset_iterators_every_epochs = 10
characters_cell_size = 64
sentence_cell_size = 300
neurons_dense_layer_after_merge = 512
classes = 4
batch_size = 256
limit_backtracking_characters = 10

CPUS = 16
size_prefetch_buffer = 10
buffer_size_shuffle = 300000
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

map_no_diac = {
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
map_correct_diac = {
	"ş": "ș",
	"Ş": "Ș",
	"ţ": "ț",
	"Ţ": "Ț",
}
map_char_to_possible_chars = {
	'a': ['ă', 'â', 'a'], 
	'i': ['î', 'i'],
	's': ['ș', 's'], 
	't': ['ț', 't']
}
map_substitute_chars = {'"': '`'}
characters_in_interest = {'a', 'i', 's', 't'}
to_lower = {}

        
def create_lower_mapping():
	for c in  string.ascii_uppercase:
		to_lower[c] = c.lower()
	to_lower['Ș'] = 'ș'
	to_lower['Â'] = 'â'
	to_lower['Ă'] = 'ă'
	to_lower['Ț'] = 'ț'
	to_lower['Î'] = 'î'

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

def bkt_all_words(index: int, clean_word: str, current_word: List) -> List:

	if index == len(clean_word):
		word = "".join(current_word)
		if word in model_embeddings.wv.vocab:
			return [word]
		else:
			return []
	else:
		L = []
		c = clean_word[index]
		if c in map_char_to_possible_chars:
			for ch in map_char_to_possible_chars[c]:
				current_word[index] = ch
				L += bkt_all_words(index + 1, clean_word, current_word)
		else:
			current_word[index] = c
			L += bkt_all_words(index + 1, clean_word, current_word)
		return L

def get_avg_possible_word(clean_word):

	with dict_lock:
		if clean_word in dict_avg_words:
			return dict_avg_words[clean_word]
		
	count_diacritics_chars = 0
	for c in clean_word:
		if c in map_char_to_possible_chars:
			count_diacritics_chars += 1
	
	if count_diacritics_chars > limit_backtracking_characters:
		return np.float32(model_embeddings.wv[clean_word])
		#return np.float32([0] * word_embedding_size)

	all_words = bkt_all_words(0, clean_word, ['a'] * len(clean_word))
	
	if len(all_words) > 0:
		return np.mean([np.float32(model_embeddings.wv[word]) for word in all_words], axis=0)
		#return np.float32([0] * word_embedding_size)
	else:
		try:
			return np.float32(model_embeddings.wv[clean_word]) 
			#return np.float32([0] * word_embedding_size)
		except:
			return np.float32([0] * word_embedding_size)

def get_embeddings_sentence(clean_tokens_sentence, index_token):
	embeddings_sentence = []

	for i in range(index_token - window_sentence, index_token + window_sentence + 1):
		if i >= 0 and i < len(clean_tokens_sentence):
			token = clean_tokens_sentence[i]
			embeddings_sentence.append(get_avg_possible_word(token))
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
def get_input_example(clean_text_utf, index_text, clean_tokens, index_sent, \
				index_last_sent, index_token):
	#global total_time_bkt
	#global total_time_sent

	# window with characters
	w = []
	for j in range(index_text - window_character, index_text + window_character + 1):
		if j < 0 or j >= len(clean_text_utf):
			v1 = padding_character
		elif ord(clean_text_utf[j]) > max_unicode_allowed:
			v1 = replace_character
		else:
			v1 = ord(clean_text_utf[j])
		w.append(v1)
	# token 
	token = clean_tokens[index_sent][index_token]
	#start_bkt = time.time()
	token_embedding = get_avg_possible_word(token)
	#end_bkt = time.time()
	#total_time_bkt = total_time + end_bkt - start_bkt
	with dict_lock:
		dict_avg_words[token] = token_embedding

	# sentence 
	#start_sen = time.time()
	# if is the same sentence don't recompute it
	if index_last_sent is None or index_sent != index_last_sent:
		sentence_embedding = get_embeddings_sentence(clean_tokens[index_sent], index_token)
	else:
		sentence_embedding = None
	#end_sen = time.time()
	#total_time_sent = total_time_sent + end_sen - start_sen

	return (np.int32(w), token_embedding, sentence_embedding)
	#return np.int32(np.array([0])), np.int32(np.array([0, 0])),  np.int32(np.array([0, 0, 0]))

def replace_char(c):

	if c in map_correct_diac:
		c = map_correct_diac[c]

	if c in to_lower:
		c = to_lower[c]

	if c in map_no_diac:
		c = map_no_diac[c]

	if ord(c) > 255:
		return chr(replace_character)
	elif c in map_substitute_chars:
		return map_substitute_chars[c]
	else:
		return c

def create_examples(original_text):
	#global total_time_tokenization
	#global total_time

	#start_all = time.time()

	original_text_utf = original_text.decode('utf-8')
	# replace some strange characters which are modified by tokenization
	clean_text_utf = "".join([replace_char(c) for c in original_text_utf])
	#start = time.time()
	clean_sentences = nltk.sent_tokenize(clean_text_utf)
	clean_tokens = []

	# construct tokens
	for i in range(len(clean_sentences)):
		clean_tokens_sent = nltk.word_tokenize(clean_sentences[i])
		clean_tokens.append(clean_tokens_sent)
	#end = time.time()

	#total_time_tokenization = total_time_tokenization + end - start

	index_text = 0 # current position in text
	index_sent = 0 # current sentence
	index_token = 0 # current token
	index_last_sent = None # last sentence computed

	# input and output lists
	window_characters = []
	word_embeddings = []
	sentence_embeddings = []
	labels = []

	while index_sent < len(clean_tokens):
		clean_token = clean_tokens[index_sent][index_token]
#		index_text = discard_first_chars(index_text, clean_text_utf, clean_token)
		i = 0		
		while i < len(clean_token):
			if clean_text_utf[index_text] in characters_in_interest:

				label = get_label(index_text, clean_text_utf, original_text_utf)
				win_char, word_emb, sent_emb = get_input_example(clean_text_utf, \
						index_text, clean_tokens, index_sent, index_last_sent, index_token)

				index_last_sent = index_sent
				window_characters.append(win_char)
				word_embeddings.append(word_emb)
				# sentence already computed
				if sent_emb is None:
					sentence_embeddings.append(sentence_embeddings[-1])
				else:
					sentence_embeddings.append(sent_emb)					
				labels.append(label)

			if clean_text_utf[index_text] == clean_token[i]:
				index_text += 1
				i += 1
			else: # discard char in text
				index_text += 1
			
		if index_token == len(clean_tokens[index_sent]) - 1:
			index_token = 0
			index_sent += 1
		else:
			index_token += 1
	# dummy values for empty sentence
	if len(window_characters) == 0:
		window_characters.append(np.int32([0] * (window_character * 2 + 1)))
		word_embeddings.append(np.float32([0] * word_embedding_size))
		sentence_embeddings.append(np.array(\
			[np.float32([0] * word_embedding_size)] * (window_sentence * 2 + 1)))
		labels.append(np.float32([0, 0, 1, 0]))

	#end = time.time()
	#total_time = total_time + end - start_all

	return (window_characters, word_embeddings, sentence_embeddings, labels)

def filter_null_strings(s):
	if len(s) == 0:
		return np.array([False])
	return np.array([True])

def flat_map_f(a, b):
	return tf.data.Dataset.from_tensor_slices((a, b))

def get_dataset(dpath, sess):

	input_files = tf.gfile.ListDirectory(dpath)
	for i in range(len(input_files)):
		input_files[i] = dpath + input_files[i]

	dataset = tf.data.TextLineDataset(input_files)

	dataset = dataset.map(lambda x: 
		tf.py_func(create_examples, (x,), (tf.int32, tf.float32, tf.float32, tf.float32), stateful=False), num_parallel_calls=CPUS)

	dataset = dataset.map(lambda x1, x2, x3, y: ((x1, x2, x3), y), num_parallel_calls=CPUS)
	dataset = dataset.flat_map(flat_map_f)
	
	dataset = dataset.shuffle(buffer_size_shuffle)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(size_prefetch_buffer)

	return dataset

with tf.Session() as sess:
	create_lower_mapping()
	dt_train = get_dataset(train_files, sess)
	dt_valid = get_dataset(valid_files, sess)
	dt_test = get_dataset(test_files, sess)

	inp_batches_train = 380968863 // batch_size
	inp_batches_test = 131424533 // batch_size
	inp_batches_valid = 131861863 // batch_size

	# inp_batches_train = 10650 // batch_size
	# inp_batches_test = 3978 // batch_size
	# inp_batches_valid = 50490 // batch_size

	# inp_batches_train = 1650 // batch_size
	# inp_batches_test = 3978 // batch_size
	# inp_batches_valid = 1490 // batch_size


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

	test_inp, test_out = iterator_test.get_next()
	test_char_window, test_words, test_sentence = test_inp
	print("char, word, sentence - char cell: {}, word cell: {}, hidden: {}".format(characters_cell_size, sentence_cell_size, neurons_dense_layer_after_merge))

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
			 steps_per_epoch=inp_batches_train//reset_iterators_every_epochs,\
			 epochs=1,\
			 verbose=1)
		
		[valid_loss, valid_acc, valid_cross_entropy] = model.evaluate([valid_char_window, valid_words, valid_sentence],\
									valid_out,\
									verbose=1,\
									steps=inp_batches_valid//reset_iterators_every_epochs)
		print("validation - loss: " + str(valid_loss) +  " acc: " + str(valid_acc))
	print("total time: " + str(total_time))
	print("tokenization time: " + str(total_time_tokenization))
	print("sentence time: " + str(total_time_sent))

	# [test_loss, test_acc, test_cross_entropy] = model.evaluate(\
	# 								[test_char_window, test_words, test_sentence],\
	# 								test_out,\
	# 								verbose=1,\
	# 								steps=inp_batches_test)
	# print("test - loss: " + str(test_loss) +  " acc: " + str(test_acc))
