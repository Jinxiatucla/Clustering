import a
import b
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import pylab as pl

r = 2
def get_svd(tfidf):
    return svds(tfidf, k = r)

data = a.retrieve_data()
tfidf = a.get_TFIDF(data)
u, s, v = get_svd(tfidf)

s = s[::-1]
r_u = (np.array(u))[:, 0:r]
r_s = s[0: r]
#print r_s
x = r_u.dot(np.diag(r_s))

y = TruncatedSVD(n_components = r)
p = y.fit_transform(tfidf)
print x 
print ""
print p
