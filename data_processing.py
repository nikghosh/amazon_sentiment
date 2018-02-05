import numpy as np
import pandas as pd
import scipy.sparse as sparse
import time

t1 = time.perf_counter()

training_df = pd.read_csv('training_data.txt', sep=' ')
test_df = pd.read_csv('test_data.txt', sep=' ')

X_train = sparse.csr_matrix(training_df.iloc[:, 1:])
y_train = training_df['Label']

X_test = sparse.csr_matrix(test_df.values)
words = training_df.columns.values[1:]

np.save('words.npy', words)

sparse.save_npz('X_train.npz', X_train)
np.save('y_train.npy', y_train)

sparse.save_npz('X_test.npz', X_test)
print(time.perf_counter() - t1)