
import os

cnt = 0
total = 0

# test_file_dir = "corpus/test/"
# valid_file_dir = "corpus/validation/"
# train_file_dir = "corpus/train/"

test_file_dir = "small_test/"
valid_file_dir = "small_valid/"
train_file_dir = "small_train/"

for filename in os.listdir(test_file_dir):
	with open(test_file_dir + filename,"r") as f:
		for line in f:
			for c in line:
				if c == "î" or c == 'ă' or c == 'â' or c == 'ș' or c == 'ț'\
				or c == 'â' or c == 'Ă' or c == 'Â' or c == 'Ș' or c == 'Ț'\
				or c == 'Â' or c == 'Î' or c == 'ş' or c == 'Ş' or c == 'ţ' or c == 'Ţ'\
				or c == 'a' or c == 'i' or c == 't' or c == 's'\
				or c == 'A' or c == 'I' or c == 'T' or c == 'S':
					cnt += 1

print("test: " + str(cnt)) 
total += cnt

cnt = 0
for filename in os.listdir(train_file_dir):
	with open(train_file_dir + filename,"r") as f:
		for line in f:
			for c in line:
				if c == "î" or c == 'ă' or c == 'â' or c == 'ș' or c == 'ț'\
				or c == 'â' or c == 'Ă' or c == 'Â' or c == 'Ș' or c == 'Ț'\
				or c == 'Â' or c == 'Î' or c == 'ş' or c == 'Ş' or c == 'ţ' or c == 'Ţ'\
				or c == 'a' or c == 'i' or c == 't' or c == 's'\
				or c == 'A' or c == 'I' or c == 'T' or c == 'S':
					cnt += 1

print("train: " + str(cnt)) 
total += cnt

cnt = 0
for filename in os.listdir(valid_file_dir):
	with open(valid_file_dir + filename,"r") as f:
		for line in f:
			for c in line:
				if c == "î" or c == 'ă' or c == 'â' or c == 'ș' or c == 'ț'\
				or c == 'â' or c == 'Ă' or c == 'Â' or c == 'Ș' or c == 'Ț'\
				or c == 'Â' or c == 'Î' or c == 'ş' or c == 'Ş' or c == 'ţ' or c == 'Ţ'\
				or c == 'a' or c == 'i' or c == 't' or c == 's'\
				or c == 'A' or c == 'I' or c == 'T' or c == 'S':
					cnt += 1

print("valid: " + str(cnt)) 
total += cnt

print("total: " + str(total))