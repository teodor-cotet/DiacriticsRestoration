
import os

cnt = 0
total = 0

for filename in os.listdir("corpus/test/"):
	with open("corpus/test/" + filename,"r") as f:
		for line in f:
			for c in line:
				if c == "î" or c == 'ă' or c == 'â' or c == 'ș' or c == 'ț'\
				or c == 'â' or c == 'Ă' or c == 'Â' or c == 'Ș' or c == 'Ț'\
				or c == 'Â' or c == 'Î' or c == 'ş' or c == 'Ş' or c == 'ţ' or c == 'Ţ':
					cnt += 1

print("test: " + str(cnt)) 
total += cnt

cnt = 0
for filename in os.listdir("corpus/train/"):
	with open("corpus/train/" + filename,"r") as f:
		for line in f:
			for c in line:
				if c == "î" or c == 'ă' or c == 'â' or c == 'ș' or c == 'ț'\
				or c == 'â' or c == 'Ă' or c == 'Â' or c == 'Ș' or c == 'Ț'\
				or c == 'Â' or c == 'Î' or c == 'ş' or c == 'Ş' or c == 'ţ' or c == 'Ţ':
					cnt += 1

print("train: " + str(cnt)) 
total += cnt

cnt = 0
for filename in os.listdir("corpus/validation/"):
	with open("corpus/validation/" + filename,"r") as f:
		for line in f:
			for c in line:
				if c == "î" or c == 'ă' or c == 'â' or c == 'ș' or c == 'ț'\
				or c == 'â' or c == 'Ă' or c == 'Â' or c == 'Ș' or c == 'Ț'\
				or c == 'Â' or c == 'Î' or c == 'ş' or c == 'Ş' or c == 'ţ' or c == 'Ţ':
					cnt += 1

print("valid: " + str(cnt)) 
total += cnt

print("total: " + str(total))