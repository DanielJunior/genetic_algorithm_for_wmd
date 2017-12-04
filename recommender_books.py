# -*- coding: utf-8 -*-

import numpy as np
import pickle
# from sklearn.neighbors import NearestNeighbors

# --- 0.0179200172424 seconds ---
#####INICIALIZAÇÃO DE VARIÁVEIS GLOBAIS!!!######
load_file = "files/book100_vec.pk"
wmd_file = "wmd.pk"
candidatos_file = "candidatos.pk"

print("Carregando arquivos...")
with open(load_file) as f:
    # X é o doc em word2vec, Bow é o bag of words, y é a classe codificada para inteiro,
    # C é o nome da classe, words são as palavras relevantes do doc
    [X, BOW_X, y, C, words] = pickle.load(f)

with open(wmd_file) as f:
    # X é o doc em word2vec, Bow é o bag of words, y é a classe codificada para inteiro,
    # C é o nome da classe, words são as palavras relevantes do doc
    WMD = pickle.load(f)

with open(candidatos_file) as f:
    # X é o doc em word2vec, Bow é o bag of words, y é a classe codificada para inteiro,
    # C é o nome da classe, words são as palavras relevantes do doc
    CANDIDATOS = pickle.load(f)


n = np.shape(X)[0]

print("Parseando dados...")
for i in xrange(n):
    X_i = X[i].T
    X_i = X_i.tolist()
    X[i] = X_i

origem_destino = [(C[i], C[j]) for i,j in enumerate([np.argpartition(WMD[i], 1)[0] for i in xrange(WMD.shape[0])])]

for docs in origem_destino:
    print("%s -> %s" % (docs[0], docs[1]))
print("FIM!")
# First let's create a dataset called X, with 6 records and 2 features each.
# X = np.array([[-1, 2], [4, -4], [-2, 1], [-1, 3], [-3, 2], [-1, 4]])

# Next we will instantiate a nearest neighbor object, and call it nbrs. Then we will fit it to dataset X.
# nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(WMD)

# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
# distances, indices = nbrs.kneighbors(X)