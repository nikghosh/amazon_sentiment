import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

import time

t1 = time.perf_counter()

with open('stem_embed_json.txt') as f:
	word_frame = pd.read_json(json.load(f), orient='split')

full_to_stem = {}
stems = list(pd.read_csv('stemmed_words.txt', sep=' '))
fulls = list(pd.read_csv('unstemmed_words.txt', sep=' '))
for i in range(len(fulls)):
	full_to_stem[fulls[i]] = stems[i]

stem_to_full = {y:x for x,y in full_to_stem.items()}

training_df = pd.read_csv('training_data.txt', sep=' ')

X_train = training_df.iloc[:, 1:]
y_train = training_df.iloc[:, 0]

dglove = 300

word_embedding = pd.DataFrame(columns=np.arange(2*dglove))
for i in range(len(y_train)):
	stem_vecs = np.zeros((dglove,1))
	row = X_train.iloc[i, :]
	criterion = row.map(lambda x: x > 0)
	present_words = list(row[criterion].axes[0].tolist())
	for stem in present_words:
		try:
			stem_vecs = np.append(stem_vecs, np.reshape(word_frame[stem_to_full[stem]], (dglove, 1)), axis=1)
		except:
			pass
	stem_vecs = stem_vecs[:, 1:]
	try:
		stem_max_vec = np.max(stem_vecs, axis=1)
		stem_min_vec = np.min(stem_vecs, axis=1)
		feat_vec = np.reshape(np.append(stem_max_vec, stem_min_vec, axis=0), (2*dglove))
	except:
		print(stem_vecs)
		feat_vec = np.zeros((2*dglove))

	word_embedding = word_embedding.append(pd.Series(feat_vec), ignore_index=True)

with open('train_embed_json_6b300d.txt', 'w') as f:
	json.dump(word_embedding.to_json(orient='split'), f)

print(time.perf_counter() - t1)