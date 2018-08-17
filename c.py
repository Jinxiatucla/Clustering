"the best r we got for LSI is 2" 

import a
import b
import pylab as pl
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer

data = a.retrieve_data()
labels = a.get_class(data)

# decomposition is the according dimension reduction function
def plot(decomposition, tfidf, r):
    truncated = decomposition(tfidf, r) 
    km = a.k_means_cluster(truncated)
    colors = map(lambda(x): 'r' if x == 0 else 'b', km.labels_)
    return truncated, colors

# set question_4_a to True to get the answer of 4(a)
if __name__ == "__main__":
    question_4_a = True
    question_4_b = True 
    print_result = False

    r_lsi = 2
    r_nmf = 2
    dataset = a.retrieve_data()
    tfidf = a.get_TFIDF(dataset)

    if question_4_a:
        # using truncated svd
        first = pl.subplot(331)
        first.set_title('truncated svd')
        truncated, colors = plot(b.get_truncated_svd, tfidf, r_lsi)
        km = a.k_means_cluster(truncated)
        if print_result:
            result = a.get_result(km, labels)
            a.print_result(result)
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

        # using nmf 
        second = pl.subplot(332)
        second.set_title('nmf')
        truncated, colors = plot(b.get_nmf, tfidf, r_lsi)
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

    # what function should we use to reduce dimension
    if question_4_b:
        # normalizing features after truncated svd
        truncated = b.get_truncated_svd(tfidf, r_lsi) 
        truncated = preprocessing.scale(truncated, with_mean = False)
        km = a.k_means_cluster(truncated)
        if print_result:
            result = a.get_result(km, labels)
            a.print_result(result)
        colors = map(lambda(x): 'r' if x == 0 else 'b', km.labels_)
        first = pl.subplot(333)
        first.set_title('normalize truncated svd festures')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

        # normalizing features after nmf 
        truncated = b.get_nmf(tfidf, r_lsi) 
        truncated = preprocessing.scale(truncated, with_mean = False)
        km = a.k_means_cluster(truncated)
        if print_result:
            result = a.get_result(km, labels)
            a.print_result(result)
        colors = map(lambda(x): 'r' if x == 0 else 'b', km.labels_)
        first = pl.subplot(334)
        first.set_title('normalize nmf festures')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)
    
        # using non-linear transformation
        non_linear = b.get_nmf(tfidf, r_nmf) 
        non_linear = FunctionTransformer(np.log1p).transform(non_linear)
        km = a.k_means_cluster(non_linear)
        if print_result:
            result = a.get_result(km, labels)
            a.print_result(result)
        colors = map(lambda(x): 'r' if x == 0 else 'b', km.labels_)
        first = pl.subplot(335)
        first.set_title('non-linear')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

        # using normalize first and then non-linear
        truncated = b.get_nmf(tfidf, r_lsi) 
        truncated = preprocessing.scale(truncated, with_mean = False)
        b_3_first = FunctionTransformer(np.log1p).transform(truncated)
        km = a.k_means_cluster(b_3_first)
        if print_result:
            result = a.get_result(km, labels)
            a.print_result(result)
        colors = map(lambda(x): 'r' if x == 0 else 'b', km.labels_)
        first = pl.subplot(336)
        first.set_title('normalize first and then non-linear')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

        # using non_linear fist and then normalize
        b_3_second = preprocessing.scale(non_linear)
        km = a.k_means_cluster(b_3_second)
        if print_result:
            result = a.get_result(km, labels)
            a.print_result(result)
        colors = map(lambda(x): 'r' if x == 0 else 'b', km.labels_)
        first = pl.subplot(337)
        first.set_title('non-linear first and then normalize')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

    pl.show()





