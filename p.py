import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD

d = np.array([[4, 2, 1], [10, 8, 3], [3, 9, 4]])
u, s, v = svd(d)
r = 1
new_s = s[0:r]
new_u = u[:, 0 : r]
new_v = v[0 : r, :]
print new_v
print new_u.dot(np.diag(new_s))

svd = TruncatedSVD(n_components = r, random_state = 42)
p = svd.fit_transform(d)
print p



