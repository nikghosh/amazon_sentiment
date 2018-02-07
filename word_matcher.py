import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

import time

t1 = time.perf_counter()
word_dict = {}
with open('glove.6B.300d.txt') as glove_6b300d:
	for entry in glove_6b300d:
		val = entry.split()
		word = val[0]
		vec = np.array(val[1:], dtype='float32', copy=False)
		word_dict[word] = vec

training_df = pd.read_csv('unstemmed_words.txt', sep=' ')
word_stems = list(training_df)
print(len(word_stems))
word_stems = [x[:-2] if x[-2] == '.' else x for x in word_stems]
print(len(word_stems))

from collections import Counter
print([k for k,v in Counter(word_stems).items() if v>1])

stem_dat = {}
lost_words = []
for stem in word_stems:
	try:
		stem_vec = word_dict[stem]
		stem_dat[stem] = stem_vec
	except:
		lost_words.append(stem)

# word_stems = lost_words
# lost_words = []

# for stem in word_stems:
# 	try:
# 		stem_vec = word_dict[stem+'e']
# 		stem_dat[stem] = stem_vec
# 	except:
# 		lost_words.append(stem)

# word_stems = lost_words
# lost_words = []

# for stem in word_stems:
# 	try:
# 		if stem[-1] == 'i':
# 			stem_vec = word_dict[stem[:-1]+'y']
# 		else: raise(ValueError)
# 		stem_dat[stem] = stem_vec
# 	except:
# 		lost_words.append(stem)

# endings = ['er', 'ing', 's', 'ence', 'ate', 'ed',
# 		   'ible', 'ic', 'y', 'ion', 'es', 'al']

# word_stems = lost_words
# lost_words = []

# for stem in word_stems:
# 	found = False
# 	for ending in endings:
# 		try:
# 			stem_vec = word_dict[stem+ending]
# 			stem_dat[stem] = stem_vec
# 			found = True
# 			break
# 		except:
# 			pass
# 	if not found:
# 		lost_words.append(stem)

# print("fourth pass", len(lost_words))
stem_dat['dissapoint'] = stem_dat['disappoint']
stem_dat['seriou'] = stem_dat['serious']

print(lost_words)

#stem_dat['youv'] = np.zeros(len(stem_dat['say']))

stem_frame = pd.DataFrame(stem_dat)
print(stem_frame)

with open('stem_embed_json.txt', 'w') as f:
	json.dump(stem_frame.to_json(orient='split'), f)

print(time.perf_counter() - t1)