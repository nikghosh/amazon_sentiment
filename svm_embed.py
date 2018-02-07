import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.decomposition import PCA

import time

t1 = time.perf_counter()

with open('train_embed_json_6b300d.txt') as f:
	train_frame = pd.read_json(json.load(f), orient='split')


train_data = pd.read_csv('training_data.txt', sep=' ')
y_train = train_data['Label']
X_bow = train_data.iloc[:, 1:]
tfidf = TfidfTransformer()
X_bow_tfidf = tfidf.fit_transform(X_bow)

pca = PCA(n_components = 100)
X_emb = train_frame.as_matrix()

trainacs = []
crossacs = []
X_emb_pca = pca.fit_transform(X_emb)
X_train = np.append(X_bow_tfidf.todense(), X_emb, axis=1)
clf = svm.LinearSVC(C = 0.11, max_iter = 45)
#X_train = X_emb
#clf = clf.fit(X_train, y_train)
#y_train_pred = clf.predict(X_train)
#train_acc = np.mean(y_train_pred == y_train)
#cross_acc = cross_val_score(clf, X_train, y_train, cv=5).mean()

# parameters = {'max_iter': range(5, 55, 5),
# 			  'C': np.arange(30) * 0.01 + 0.01
			  
# }

# gs_clf = GridSearchCV(clf, parameters, n_jobs=-1, cv=5)
# gs_clf = gs_clf.fit(X_train, y_train)


print(gs_clf.best_score_)
for param_name in sorted(parameters.keys()):
	print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
# for i in range(10):
# 	pca = PCA(n_components = 10*(i+1))
# 	X_emb_pca = pca.fit_transform(X_emb)
# 	X_train = X_emb_pca
# 	clf = svm.LinearSVC(C = 0.1, max_iter = 50)
# 	clf = clf.fit(X_train, y_train)
# 	y_train_pred = clf.predict(X_train) 
# 	train_acc = np.mean(y_train_pred == y_train)
# 	cross_acc = cross_val_score(clf, X_train, y_train, cv=5).mean()
# 	trainacs.append(train_acc)
# 	crossacs.append(cross_acc)
print(train_acc)
print(cross_acc)