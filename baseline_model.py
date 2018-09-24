import nltk
import os
import pickle
from typing import List
import string

test_file_dir = "corpus/test/"
valid_file_dir = "corpus/validation/"
train_file_dir = "corpus/train/"
small_file_dir = "small/"

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
to_lower = {}

	
def create_lower_mapping():
	for c in  string.ascii_uppercase:
		to_lower[c] = c.lower()
	to_lower['Ș'] = 'ș'
	to_lower['Â'] = 'â'
	to_lower['Ă'] = 'ă'
	to_lower['Ț'] = 'ț'
	to_lower['Î'] = 'î'
	return to_lower

def get_clean_word_with_diacs(w):
	clean_word = ['a'] * len(w)
	for i in range(len(clean_word)):
		c = w[i]
		clean_word[i] = c
		if c in map_correct_diac:
			clean_word[i] = map_correct_diac[c]
		if c in to_lower:
			clean_word[i] = to_lower[c]
	return "".join(clean_word)

def get_clean_word(w):
	global to_lower
	clean_word = ['a'] * len(w)
	for i in range(len(clean_word)):
		c = w[i]
		clean_word[i] = c
		if c in map_correct_diac:
			clean_word[i] = map_correct_diac[c]
		if c in to_lower:
			clean_word[i] = to_lower[c]
		if c in map_no_diac:
			clean_word[i] = map_no_diac[c]
	return "".join(clean_word)

def get_clean_word_for_comparison(w):
	global to_lower
	clean_word = ['a'] * len(w)
	for i in range(len(clean_word)):
		c = w[i]
		clean_word[i] = c
		if c in map_correct_diac:
			clean_word[i] = map_correct_diac[c]
		if c in to_lower:
			clean_word[i] = to_lower[c]
	return "".join(clean_word)
	
def bkt_all_words(index: int, clean_word: str, current_word: List) -> List:

	if index == len(clean_word):
		word = "".join(current_word)
		return [word]
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



def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def choose_highest_occurrence_word(train_words, all_words):

	highest_word = None
	for w in all_words:
		if w in all_words:
			if w in train_words and highest_word is None:
				highest_word = w
			elif highest_word in train_words and w in train_words and train_words[highest_word] < train_words[w]:
				highest_word = w
	if highest_word is None:
		highest_word = all_words[0]
	return highest_word

def count_possible_characters(clean_word):
	count_diacritics_chars = 0
	for c in clean_word:
		if c in map_char_to_possible_chars:
			count_diacritics_chars += 1
	return count_diacritics_chars

def get_words(direct):
	train_words = {}
	for filename in os.listdir(direct):
		with open(direct + filename,"r") as f:
			for line in f:
				clean_sentences = nltk.sent_tokenize(line)
				if clean_sentences is None:
					continue
				for i in range(len(clean_sentences)):
					clean_tokens_sent = nltk.word_tokenize(clean_sentences[i])
					
					if clean_tokens_sent is None:
						continue

					for w in clean_tokens_sent:
						if w is not None:
							w = get_clean_word_with_diacs(w)
							if w not in train_words:
								train_words[w] = 1
							else:
								train_words[w] += 1
	return train_words

def test_words(train_words, direct):
	correct = 0
	wrong = 0
	cnt_large = 0
	for filename in os.listdir(direct):
		with open(direct + filename, "r") as f:
			for line in f:
				clean_sentences = nltk.sent_tokenize(line)
				if clean_sentences is None:
					continue
				for i in range(len(clean_sentences)):
					clean_tokens_sent = nltk.word_tokenize(clean_sentences[i])
					if clean_tokens_sent is None:
						continue
					for w in clean_tokens_sent:

						if w is None:
							continue

						clean_word = get_clean_word(w)
						clean_word_comp = get_clean_word_for_comparison(w)
						cnt = count_possible_characters(clean_word)
						if cnt < 12 and cnt > 0:
							all_words = bkt_all_words(0, clean_word, ['a'] * len(clean_word))
							#print(all_words)
							highest_common_word = choose_highest_occurrence_word(train_words, all_words)
							#print(highest_common_word, clean_word_comp)
							if highest_common_word == clean_word_comp:
								correct += 1
							else:
								wrong += 1
						else:
							cnt_large += 1
	print('missed: ' + str(cnt_large))
	acc = correct / (correct + wrong)
	print('acc words: ' + str(acc))
	return acc				

if __name__ == '__main__':
	to_lower = create_lower_mapping()
	#train_words_stats = get_words(train_file_dir)
	name = "words_train_rowiki_stats"
	#save_obj(train_words_stats, name)
	train_words_stats = load_obj(name)
	acc = test_words(train_words_stats, test_file_dir)
	print(acc)
	#print(get_clean_word_with_diacs('ășasțțâăî'))
