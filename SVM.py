import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm

X_train = sparse.load_npz('X_train.npz')
y_train = np.load('y_train.npy')


text_clf = Pipeline([('tfidf', TfidfTransformer()),
					 ('clf', svm.LinearSVC(C = 0.1, max_iter = 10)),
	])

text_clf = text_clf.fit(X_train, y_train)
y_train_pred = text_clf.predict(X_train) 
train_acc = np.mean(y_train_pred == y_train)

print(train_acc)

print(cross_val_score(text_clf, X_train, y_train, cv=5).mean())

'''
parameters = {'clf__max_iter': (15, 10, 20),
			  'clf__C': (0.05, 0.1, 0.15),
			  'tfidf__use_idf': (True, False)
			  
}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=5)
gs_clf = gs_clf.fit(X_train, y_train)


print(gs_clf.best_score_)
for param_name in sorted(parameters.keys()):
	print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
'''

# print('elapsed time: ' + str(end_time - start_time))

write_results = True

if write_results:
	X_test = sparse.load_npz('X_test.npz')
	y_test_pred = text_clf.predict(X_test)
	with open('submit1.txt', 'w') as f:
		f.write('Id,Prediction\n')
		for i in range(len(y_test_pred)):
			f.write(str(i+1) + ',' + str(y_test_pred[i]) + '\n')

	f.close()
