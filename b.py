import a
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import pylab as pl

r = [1, 2, 3, 5, 10, 20, 50, 100, 300]
data = a.retrieve_data()

# plot the variance v.s. r
def get_svd(tfidf):
    number = 1000
    U, s, V = svds(tfidf, number)
#    print tfidf.shape
#    print U.shape 
#    print s.shape 
#    print V.shape 
    total = np.trace((tfidf.dot(np.transpose(tfidf))).toarray())
    s = s[::-1]
    s = map(lambda(x): x * x, s)
    for i in range(1, number):
        s[i] += s[i - 1]
    s = map(lambda(x): x / total, s)
    x = range(1, number + 1)
    print s[number - 1]
    pl.plot(x, s)
    pl.show()
    return U, s, V

# get nmf
def get_nmf(tfidf, rank):
    model = NMF(n_components = rank)
    return model.fit_transform(tfidf)

# get the truncated svd
def get_truncated_svd(tfidf, rank):
    svd = TruncatedSVD(n_components = rank)
    return svd.fit_transform(tfidf)

# use LSI to do the reduction
# u, s, v are not in use
# reduction_type is a function to determine which reduction method is used,get_truncated_svd or get_nmf
def reduce_plot(U, s, V, reduction_type, tfidf, k = 2):
    U = np.array(U) 
    V = np.array(V) 
    s = np.array(s) 
    labels = a.get_class(data)
    homo_list = []
    complete_list = []
    vscore_list = []
    rand_list = []
    mutual_list = []
    for rank in r:
        # get the reducted U, s, V
#        r_U = U[:, 0:rank] 
#        r_s = s[0:rank]
#        x = r_U.dot(np.diag(r_s))
        x = reduction_type(tfidf, rank) 
        km = a.k_means_cluster(x, k) 
        result = a.get_result(km, labels)
        homo_list.append(result[0])
        complete_list.append(result[1])
        vscore_list.append(result[2])
        rand_list.append(result[3])
        mutual_list.append(result[4])
    # plot
    pl.plot(r, homo_list, label = "homo")
    pl.plot(r, complete_list, label = "complete")
    pl.plot(r, vscore_list, label = "vscore")
    pl.plot(r, rand_list, label = "rand")
    pl.plot(r, mutual_list, label = "mutual")
    pl.legend(loc = "upper right")
    pl.show()

if __name__ == "__main__":
    data = a.retrieve_data()
    tfidf = a.get_TFIDF(data)
    U, s, V = get_svd(tfidf)
   # reduce_plot(U, s, V, get_truncated_svd, tfidf)
    reduce_plot(U, s, V, get_nmf, tfidf)
