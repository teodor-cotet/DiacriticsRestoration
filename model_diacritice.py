import tensorflow as tf
import numpy as np
from tensorflow import keras
import nltk
from gensim.models.wrappers import FastText as FastTextWrapper

import unidecode
import re
import string
import time
from typing import List

import argparse
import os
import threading

args = None
model_embeddings = None

dict_lock = threading.Lock()
dict_avg_words = {}

# the weights of the model will be saved in a folder per each epoch
folder_saved_models = "saved_models/"
fast_text = "fastText/wiki.ro"

window_sentence = 15 
window_character = 6 # 2 * x + 1

character_embedding_size = 20
word_embedding_size = 300


characters_cell_size = 64
sentence_cell_size = 300
neurons_dense_layer_after_merge = 512
batch_size = 4096
limit_backtracking_characters = 10

CPUS = 7
size_prefetch_buffer = 10
max_unicode_allowed = 770
replace_character = 255
padding_character = 0

train_files = "corpus/train/"
test_files = "corpus/test/"
valid_files = "corpus/validation/"

samples_number = {
	'full_train': 380968863,
	'full_test': 131424533,
	'full_valid': 131861863,
	'par_train': 84321880,
	'par_test': 29407143,
	'par_valid': 28882058,
}

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

# get case of the highest probability (returned by softmax)
def get_case(p):
	case = 0
	for i in range(len(p)):
		if p[i] > p[case]:
			case = i
	return case

def create_lower_mapping():
	for c in  string.ascii_uppercase:
		to_lower[c] = c.lower()
	to_lower['Ș'] = 'ș'
	to_lower['Â'] = 'â'
	to_lower['Ă'] = 'ă'
	to_lower['Ț'] = 'ț'
	to_lower['Î'] = 'î'

# case:
#	0 -> ă
#	1 -> â, î
#	2 -> unmodified
# 	3 -> ș, ț
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
			if args.nr_classes == 5:
				case = 4
			else:
				case = 3
		elif original_text_utf[i] == 's':
			case = 2

	label = np.float32([0] * args.nr_classes)
	label[case] = np.float32(1.0)
	return label

def bkt_all_words(index: int, clean_word: str, current_word: List) -> List:

	if index == len(clean_word):
		word = "".join(current_word)
		if args.use_dummy_word_embeddings == True:
			return [word]
		else:
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
		if args.use_dummy_word_embeddings == False:
			return np.float32(model_embeddings.wv[clean_word])
		else:
			return np.float32([0] * word_embedding_size)

	all_words = bkt_all_words(0, clean_word, ['a'] * len(clean_word))
	
	if len(all_words) > 0:
		if args.use_dummy_word_embeddings == False:
			return np.mean([np.float32(model_embeddings.wv[word]) for word in all_words], axis=0)
		else:
			return np.float32([0] * word_embedding_size)
	else:
		try:
			if args.use_dummy_word_embeddings == False:
				return np.float32(model_embeddings.wv[clean_word])
			else:
				return np.float32([0] * word_embedding_size)
		except:
			return np.float32([0] * word_embedding_size)

def get_embeddings_sentence(clean_tokens_sentence, index_token):
	embeddings_sentence = []

	for i in range(index_token - window_sentence, index_token + window_sentence + 1):
		if i >= 0 and i < len(clean_tokens_sentence):
			token = clean_tokens_sentence[i]
			token_embedding = get_avg_possible_word(token)
			embeddings_sentence.append(token_embedding)
			with dict_lock:
				dict_avg_words[token] = token_embedding
		else:
			embeddings_sentence.append(np.float32([0] * word_embedding_size))
	return np.array(embeddings_sentence)

# return an tuple input (window_char, embedding_token, embedding_sentence)
def get_input_example(clean_text_utf, index_text, clean_tokens, index_sent, \
				index_last_sent, index_token):

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
	token_embedding = get_avg_possible_word(token)
	with dict_lock:
		dict_avg_words[token] = token_embedding

	# sentence 
	# if is the same sentence don't recompute it
	if index_last_sent is None or index_sent != index_last_sent:
		sentence_embedding = get_embeddings_sentence(clean_tokens[index_sent], index_token)
	else:
		sentence_embedding = None

	return (np.int32(w), token_embedding, sentence_embedding)

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

def replace_char_original(c):
	if c in map_correct_diac:
		c = map_correct_diac[c]

	if c in to_lower:
		c = to_lower[c]
	return c

def count_chars_in_interest(s):
	cnt_chars = 0
	chars_in_int = []
	for c in s:
		character_c = chr(c)
		if character_c in characters_in_interest:
			cnt_chars += 1
			chars_in_int.append(character_c)
	return cnt_chars, chars_in_int

def create_examples(original_text, is_test_dataset):
	drop_example = False

	try:
		original_text_utf = original_text.decode('utf-8')
		original_text_utf = "".join([replace_char_original(c) for c in original_text_utf])
		# replace some strange characters which are modified by tokenization
		clean_text_utf = "".join([replace_char(c) for c in original_text_utf])
		clean_sentences = nltk.sent_tokenize(clean_text_utf)
		clean_tokens = []

		# construct tokens
		for i in range(len(clean_sentences)):
			clean_tokens_sent = nltk.word_tokenize(clean_sentences[i])
			clean_tokens.append(clean_tokens_sent)

		index_text = 0 # current position in text
		index_sent = 0 # current sentence
		index_token = 0 # current token
		index_last_sent = None # last sentence computed

		# input and output lists
		clean_words = []
		window_characters = []
		word_embeddings = []
		sentence_embeddings = []
		labels = []
		while index_sent < len(clean_tokens):
			clean_token = clean_tokens[index_sent][index_token]
			i = 0		
			while i < len(clean_token):

				if clean_text_utf[index_text] in characters_in_interest:

					label = get_label(index_text, clean_text_utf, original_text_utf)
					#print(original_text_utf[index_text], label)
					win_char, word_emb, sent_emb = get_input_example(clean_text_utf, \
							index_text, clean_tokens, index_sent, index_last_sent, index_token)
					index_last_sent = index_sent
					if is_test_dataset == True:
						clean_words.append(clean_token)
					window_characters.append(win_char)
					word_embeddings.append(word_emb)
					# sentence already computed
					if sent_emb is None:
						sentence_embeddings.append(sentence_embeddings[-1])
					else:
						sentence_embeddings.append(sent_emb)					
					labels.append(label)
					#print(clean_text_utf[index_text], original_text_utf[index_text], label)

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
	except:
		drop_example = True
			
			
	# dummy values for empty sentence
	if len(window_characters) == 0 or drop_example == True:
		clean_words = ['a']
		window_characters = [np.int32([0] * (window_character * 2 + 1))]
		word_embeddings = [np.float32([0] * word_embedding_size)]
		sentence_embeddings = [np.array(\
			[np.float32([0] * word_embedding_size)] * (window_sentence * 2 + 1))]
		lab = np.float32([0] * args.nr_classes)
		lab[2] = np.float32(1.0)
		labels = [lab]

	if is_test_dataset == False:
		return (window_characters, word_embeddings, sentence_embeddings, labels)
	else:
		return (clean_words, window_characters, word_embeddings, sentence_embeddings, labels)

def filter_null_strings(s):
	if len(s) == 0:
		return np.array([False])
	return np.array([True])

def flat_map_f(a, b):
	return tf.data.Dataset.from_tensor_slices((a, b))

def get_dataset(dpath, sess, is_test_dataset=False):

	input_files = tf.gfile.ListDirectory(dpath)
	
	for i in range(len(input_files)):			
		if args.corpus_rowiki == False and input_files[i].count('rowiki') > 0:
			input_files.remove(input_files[i])
			break
	
	for i in range(len(input_files)):
		input_files[i] = dpath + input_files[i]

	dataset = tf.data.TextLineDataset(input_files)
	if is_test_dataset == True:
		datatype_returned = (tf.string, tf.int32, tf.float32, tf.float32, tf.float32)
	else:
		datatype_returned = (tf.int32, tf.float32, tf.float32, tf.float32)

	dataset = dataset.map(lambda x:\
		tf.py_func(create_examples,\
					(x,is_test_dataset,),\
					datatype_returned,\
					stateful=False),\
		num_parallel_calls=CPUS)
	if is_test_dataset == True:
		dataset = dataset.map(lambda x1, x2, x3, x4, y:\
								((x1, x2, x3, x4), y),\
								num_parallel_calls=CPUS)
	else:
		dataset = dataset.map(lambda x1, x2, x3, y:\
								((x1, x2, x3), y),\
								num_parallel_calls=CPUS)
	dataset = dataset.flat_map(flat_map_f)
	
	# do not shuffle or batch test dataset
	if is_test_dataset == True:
		dataset = dataset.batch(batch_size, drop_remainder=True)
	else:
		dataset = dataset.shuffle(args.buffer_size_shuffle)
		dataset = dataset.batch(batch_size)
		dataset = dataset.prefetch(size_prefetch_buffer)

	return dataset

def compute_prediction(correct_case, simple_c, prediction_chars, correct=0):
	# correct 1 - true, 0 - false
	if simple_c == 'a':
		if correct_case == 0: # ă
			prediction_chars['ă'][correct] += 1
		elif correct_case == 1: # 
			prediction_chars['â'][correct] += 1
		elif correct_case == 2:
			prediction_chars['a'][correct] += 1
	elif simple_c == 'i':
		if correct_case == 1:
			prediction_chars['î'][correct] += 1
		elif correct_case == 2:
			prediction_chars['i'][correct] += 1
	elif simple_c == 't':
		if correct_case == 3:
			prediction_chars['ț'][correct] += 1
		elif correct_case == 2:
			prediction_chars['t'][correct] += 1
	elif simple_c == 's':
		if correct_case == 3 or correct_case == 4:
			prediction_chars['ș'][correct] += 1
		elif correct_case == 2:
			prediction_chars['s'][correct] += 1

	
def compute_test_accuracy(sess, model):
	dt_test = get_dataset(test_files, sess, True)
	iterator_test = dt_test.make_initializable_iterator()

	sess.run(iterator_test.initializer)
	nr_test_batches = args.number_samples_test

	test_inp_pred, test_out_pred = iterator_test.get_next()
	test_string_word_pred, test_char_window_pred, test_words_pred, test_sentence_pred = test_inp_pred
	#  test_words_pred, test_sentence_pred],\
	predictions = model.predict(x=test_char_window_pred,
				  verbose=1,
				  steps=nr_test_batches)
	print(len(predictions))
	current_test_batch = 0
	sess.run(iterator_test.initializer)
	test_next_element = iterator_test.get_next()
	prediction_index = 0
	total_words = 0
	correct_predicted_words = 0
	correct_predicted_chars = 0
	wrong_predicatd_chars = 0
	prediction_chars = {'ă': [0, 0], 'â': [0, 0], 'a': [0, 0], 'i': [0, 0], 'î': [0, 0], 's': [0, 0],\
						'ș': [0, 0], 't': [0, 0], 'ț': [0, 0]}
	# prediction_chars[x] means that it should have been x, but it something else

	wrong_restoration_words  = {}
	correct_restoration_words = {}
	acc_restoration_word = {}
	all_words= set()

	while True:
		try:
			test_inp, test_out = sess.run(test_next_element)
			test_string_word, _, _, _ = test_inp
			current_test_batch += 1
			index_batch = 0
			exit_batch = False
			#print('batch {} out of {}'.format(current_test_batch, nr_test_batches))
			while index_batch < len(test_string_word):
				# skip last word no matter what
				word = test_string_word[index_batch]
				all_words.add(word)
				nr_chars_in_word, chars_in_int = count_chars_in_interest(word)
				if nr_chars_in_word > len(test_string_word) - index_batch:
					prediction_index += len(test_string_word) - index_batch
					break

				correct_prediction_word = True
				for i in range(nr_chars_in_word):
					pred_vector = predictions[prediction_index]
					predicted_case = get_case(pred_vector)
					correct_case = get_case(test_out[index_batch])
					index_batch += 1
					prediction_index += 1

					if predicted_case != correct_case:
						correct_prediction_word = False
						compute_prediction(correct_case, chars_in_int[i], prediction_chars, correct=0)
						wrong_predicatd_chars += 1
					else:
						compute_prediction(correct_case, chars_in_int[i], prediction_chars, correct=1)
						correct_predicted_chars += 1
				
				total_words += 1

				if correct_prediction_word == True:
					correct_predicted_words += 1
					if word in correct_restoration_words:
						correct_restoration_words[word] += 1
					else:
						correct_restoration_words[word] = 1
				else:
					if word in wrong_restoration_words:
						wrong_restoration_words[word] += 1
					else:
						wrong_restoration_words[word] = 1
					
			if current_test_batch == nr_test_batches:
				break
		except tf.errors.OutOfRangeError:
			break

	index_word = 0
	print('highest missed words: ')
	for key, value in sorted(wrong_restoration_words.items(), key=lambda x: x[1], reverse=True):
		correct = 0
		if key in correct_restoration_words:
			correct = correct_restoration_words[key]
		print("word '"  + key.decode('utf-8') + "' wrong: " +  str(value) + \
			' correct: ' + str(correct) + ' accuracy: ' + str(1.0 * correct / (value + correct)))
		index_word += 1
		if index_word == args.top_wrong_words_restoration:
			break

	print('accuracy per characters: ')

	for key, values in prediction_chars.items():
		if values[0] + values[1] != 0:
			print(key + ': ' + str(values[1] / (values[0] + values[1])))

	(char_acc, word_acc) = (correct_predicted_chars / (correct_predicted_chars + wrong_predicatd_chars), correct_predicted_words / total_words)
	print("char acc: " + str(char_acc) + ", word accuracy: " + str(word_acc) + ' ')
	return char_acc, word_acc

def set_up_folders_saved_models():
	full_path_dir = folder_saved_models + args.folder_saved_model_per_epoch
	if args.save_model == True:
		if os.path.exists(full_path_dir) == False:
			os.makedirs(full_path_dir)
		elif args.load_model_name is None:
			print('a folder with the same name (' + args.folder_saved_model_per_epoch +\
				 ') for saving model already exists, delete it to continue or give other name to the folder of the saved model')
			exit(0)

def parse_args():
	global args
	parser = argparse.ArgumentParser(description='Run diacritics model')
	parser.add_argument('-s', dest="save_model", action='store_false', default=True,\
						help="save the model (and weights), default=true")
	parser.add_argument('-f', dest="folder_saved_model_per_epoch",\
						action='store', default="char_word_sentence",\
						help="name of the folder to store the weights, default: char_word_sentence")
	parser.add_argument('-c', dest="corpus_rowiki",\
						action='store_true', default=False,\
						help="if you want to use rowiki corpus, beside parliament corpus, default=false")
	parser.add_argument('-test', dest="do_test",\
						action='store_true', default=False,\
						help="if you want to run test dataset, default=false")
	parser.add_argument('-n_test', dest="number_samples_test",\
						action='store', default=samples_number['par_test'] // batch_size, type=int,\
						help="number of samples for test accuracy, if -test is not set \
						this does not have any effect, default=100000")
	parser.add_argument('-e', dest="epochs",\
						action='store', default=20, type=int,\
						help="number of epochs, default=20")
	parser.add_argument('-r', dest="reset_iterators_every_epochs",\
						action='store', default=10, type=int,\
						help="reset the iterators for the dataset every nr epochs, default=10")
	parser.add_argument('-buff', dest="buffer_size_shuffle",\
						action='store', default=100000, type=int,\
						help="size of the buffer for shuffle, default=100000")
	parser.add_argument('-fastt', dest="use_dummy_word_embeddings",\
						action='store_true', default=False,\
						help="use dummy word embeddings instead of fasttext, default=false")
	parser.add_argument('-load', dest="load_model_name",\
						action='store', default=None,\
						help="load presaved model and weights\
						, specify just the folder name, it will take the last epoch file,\
						default=None")
	parser.add_argument('-tv', dest="run_train_validation",\
						action='store_false', default=True,\
						help="run train and validation, if false you should set -load model param\
						, default=True")
	parser.add_argument('-mgpu', dest="percent_memory_gpu",\
						action='store', default=0.2, type=float,\
						help="percentage of the gpu memory to use, default=0.2")
	parser.add_argument('-wrong', dest="top_wrong_words_restoration",\
						action='store', default=30, type=int,\
						help="hardest words to restore, default=30")
	parser.add_argument('-char', dest="use_window_characters",\
						action='store_false', default=True,\
						help="if model should use window of characters, default=True")
	parser.add_argument('-word', dest="use_word_embedding",\
						action='store_false', default=True,\
						help="if model should use word embeddings, default=True")
	parser.add_argument('-sent', dest="use_sentence_embedding",\
						action='store_false', default=True,\
						help="if model should use sentence embedding, default=True")
	parser.add_argument('-hidden', dest="hidden_neurons", required=True,\
						action='append', type=int,\
						help="number of neurons on the hidden layer, no default")
	parser.add_argument('-classes', dest="nr_classes", default=4,\
						action='store', type=int,\
						help="number of classes to be used (4 or 5), default=4")
	args = parser.parse_args()
	args.folder_saved_model_per_epoch += '/'
	if args.load_model_name is not None:
		args.load_model_name += '/'

	if args.nr_classes != 4 and args.nr_classes != 5:
		print('classes has to be either 4 or 5, exit')
		exit(0)

	for k in args.__dict__:
		if args.__dict__[k] is not None:
			print(k, '->', args.__dict__[k])

def get_number_samples():
	global args
	if args.corpus_rowiki == False:
		inp_batches_train = samples_number['par_train'] // batch_size
		inp_batches_test = samples_number['par_test'] // batch_size
		inp_batches_valid = samples_number['par_valid'] // batch_size
	else:
		inp_batches_train = samples_number['full_train'] // batch_size
		inp_batches_test = samples_number['full_test'] // batch_size
		inp_batches_valid = samples_number['full_valid'] // batch_size
	return inp_batches_train, inp_batches_test, inp_batches_valid

def get_input_list(characters_bi_lstm_layer, word_embeddings_layer, sentence_bi_lstm_layer):
	input_list = []
	if args.use_window_characters == True:
		input_list.append(characters_bi_lstm_layer)

	if args.use_word_embedding == True:
		input_list.append(word_embeddings_layer)

	if args.use_sentence_embedding == True:
		input_list.append(sentence_bi_lstm_layer)
	return input_list

# construct the model 
def construct_model(sess):
	
	last_epoch = 0
	if args.load_model_name is not None:
		
		folder_path_with_epochs = folder_saved_models + args.load_model_name 
		epochs_files = os.listdir(folder_path_with_epochs)
		sorted_epochs_files = sorted(epochs_files)
		load_file = sorted_epochs_files[-1]
		print('loading model from: ' + folder_saved_models +\
		 		args.load_model_name + load_file)
		model = keras.models.load_model(folder_saved_models + args.load_model_name + load_file)
		last_epoch = len(sorted_epochs_files)
	else:
		vocabulary_size = max_unicode_allowed + 1
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
		# merged
		input_list = get_input_list(characters_bi_lstm_layer,\
			word_embeddings_layer, sentence_bi_lstm_layer)
		if len(input_list) > 1:
			merged_layer = keras.layers.concatenate(input_list, axis=-1)
		else:
			merged_layer = input_list[0]
		prev_layer = merged_layer
		# hidden layers
		for h_neurons in args.hidden_neurons:
			prev_layer = keras.layers.Dense(h_neurons, activation='tanh')(prev_layer)
		output = keras.layers.Dense(args.nr_classes, activation='softmax')(prev_layer)

		model = keras.models.Model(inputs=[input_character_window, word_embeddings_layer, sentence_embeddings_layer],\
								outputs=output)
		model.compile(optimizer='adam',\
					loss='categorical_crossentropy',\
					metrics=['accuracy'])
	return model, last_epoch

if __name__ == "__main__":

	create_lower_mapping()
	parse_args()
	if args.load_model_name is not None:
		print('loading last epoch weights from ' + args.load_model_name)

	inp_batches_train, inp_batches_test, inp_batches_valid = get_number_samples()
	set_up_folders_saved_models()

	if args.use_dummy_word_embeddings == False:
		model_embeddings = FastTextWrapper.load_fasttext_format(fast_text)
	
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = args.percent_memory_gpu

	with tf.Session(config=config) as sess:

		if inp_batches_test < args.number_samples_test:
			print("cannot start, too many test samples given, has to be lower than "\
				 + str(inp_batches_test))

		print("char, word, sentence - char cell:{}, word cell: {}, hidden: {}"\
			.format(characters_cell_size, sentence_cell_size, neurons_dense_layer_after_merge))

		model, last_epoch = construct_model(sess)
		# run test and validation 
		if args.run_train_validation == True:

			dt_train = get_dataset(train_files, sess)
			dt_valid = get_dataset(valid_files, sess)
			iterator_train = dt_train.make_initializable_iterator()
			iterator_valid = dt_valid.make_initializable_iterator()

			for i in range(last_epoch, last_epoch + args.epochs):

				print('epoch: ' + str(i + 1))
				# reset iterators
				if (i - last_epoch) % args.reset_iterators_every_epochs == 0:
					print('resseting iterators')
					sess.run(iterator_valid.initializer)
					valid_inp, valid_out = iterator_valid.get_next()
					valid_char_window, valid_words, valid_sentence = valid_inp

					sess.run(iterator_train.initializer)
					train_inp, train_out = iterator_train.get_next()
					train_char_window, train_words, train_sentence = train_inp
				
				# train an epoch
				model.fit(\
					[train_char_window, train_words, train_sentence],\
					[train_out],\
					steps_per_epoch=inp_batches_train//args.reset_iterators_every_epochs,\
					epochs=1,\
					verbose=1)

				# save weights
				if args.save_model == True:
					print('saving model (and weights)')
					full_path_epoch_weights = folder_saved_models + args.folder_saved_model_per_epoch +\
							'epoch_' + str(i) + '.h5'
					model.save(full_path_epoch_weights)
				# validate 
				[valid_loss, valid_acc] = model.evaluate([valid_char_window, valid_words, valid_sentence],\
											valid_out,\
											verbose=1,\
											steps=inp_batches_valid//args.reset_iterators_every_epochs)
				print("validation - loss: " + str(valid_loss) +  " acc: " + str(valid_acc))
		# test
		if args.do_test:
			compute_test_accuracy(sess, model)