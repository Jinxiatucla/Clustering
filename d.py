"the best r of nmf and lsi are both 10"

import a
import b
import c
from sklearn.datasets import fetch_20newsgroups
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer 
import numpy as np
import pylab as pl

r = [1, 2, 3, 5, 10, 20, 50, 100, 300]
r_lsi = 10
r_nmf = 10
k = 20
all_colors = ['silver', 'firebrick', 'sandybrown', 'tan', 'gold', 'lightseagreen', 'deepskyblue', 'darkorchid', 'palevioletred', 'slateblue', 'fuchsia', 'pink', 'y', 'olive', 'c', 'black', 'g', 'r', 'yellow', 'aqua', 'gray']

def get_result(reduce_function, tfidf, labels):
    homo_list = []
    complete_list = []
    vscore_list = []
    rand_list = []
    mutual_list = []
    # using truncated svd
    for rank in r:
        reduced = reduce_function(tfidf, rank)
        km = a.k_means_cluster(reduced, k)
        result = a.get_result(km, labels)
        homo_list.append(result[0])
        complete_list.append(result[1])
        vscore_list.append(result[2])
        rand_list.append(result[3])
        mutual_list.append(result[4])
    pl.plot(r, homo_list, label = "homo")
    pl.plot(r, complete_list, label = "complete")
    pl.plot(r, vscore_list, label = "vscore")
    pl.plot(r, rand_list, label = "rand")
    pl.plot(r, mutual_list, label = "mutual")
    pl.legend(loc = "upper right")
    pl.show()


# plot the 20 clusters
def plot(decomposition, tfidf, r):
    truncated = decomposition(tfidf, r)
    km = a.k_means_cluster(truncated, k)
    c = map(lambda(x): all_colors[x], km.labels_)
    return truncated, c

if __name__ == "__main__":
    dataset = fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 42) 
    tfidf = a.get_TFIDF(dataset)
    labels = dataset.target
    # uncomment the next two lines to get the plot of lsi and nmf
#    get_result(b.get_truncated_svd, tfidf)
#    get_result(b.get_nmf, tfidf, labels)
    
    question_5_a = True
    question_5_b = True
    print_result = False
    
    if question_5_a:
        # get the 20-color clusters 
        # using truncated svd
        aa = pl.subplot(331)
        aa.set_title("truncated svd")
        truncated, colors = plot(b.get_truncated_svd, tfidf, r_lsi)
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

        # using nmf
        aa = pl.subplot(332)
        aa.set_title("nmf")
        truncated, colors = plot(b.get_nmf, tfidf, r_nmf)
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

    # what function should we use to reduce dimension
    if question_5_b:
        # normalizing features after truncated svd 
        truncated = b.get_truncated_svd(tfidf, r_nmf) 
        truncated = preprocessing.scale(truncated, with_mean = False)
        km = a.k_means_cluster(truncated, k)
        if print_result:
            result = a.get_result(km, labels)
            a.print_result(result)
        colors = map(lambda(x): all_colors[x], km.labels_)
        first = pl.subplot(333)
        first.set_title('normalize festures using truncated svd')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

        # normalizing features after nmf
        truncated = b.get_nmf(tfidf, r_nmf) 
        truncated = preprocessing.scale(truncated, with_mean = False)
        km = a.k_means_cluster(truncated, k)
        if print_result:
            result = a.get_result(km, labels)
            a.print_result(result)
        colors = map(lambda(x): all_colors[x], km.labels_)
        first = pl.subplot(334)
        first.set_title('normalize festures using nmf')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)
    
        # using non-linear transformation
        non_linear = b.get_nmf(tfidf, r_nmf) 
        non_linear = FunctionTransformer(np.log1p).transform(non_linear)
        km = a.k_means_cluster(non_linear, k)
        if print_result:
            result = a.get_result(km, labels)
            a.print_result(result)
        colors = map(lambda(x): all_colors[x], km.labels_)
        first = pl.subplot(335)
        first.set_title('non-linear')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

        # using normalize first and then non-linear
        truncated = b.get_nmf(tfidf, r_lsi) 
        truncated = preprocessing.scale(truncated, with_mean = False)
        b_3_first = FunctionTransformer(np.log1p).transform(truncated)
        km = a.k_means_cluster(b_3_first, k)
        if print_result:
            result = a.get_result(km, labels)
            a.print_result(result)
        colors = map(lambda(x): all_colors[x], km.labels_)
        first = pl.subplot(336)
        first.set_title('normalize first and then non-linear')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

        # using non_linear fist and then normalize
        b_3_second = preprocessing.scale(non_linear)
        km = a.k_means_cluster(b_3_second, k)
        if print_result:
            result = a.get_result(km, labels)
            a.print_result(result)
        colors = map(lambda(x): all_colors[x], km.labels_)
        first = pl.subplot(337)
        first.set_title('non-linear first and then normalize')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

    pl.show()









