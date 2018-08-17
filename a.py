from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import homogeneity_score,completeness_score, adjusted_rand_score, adjusted_mutual_info_score
import sklearn.metrics as metrics
import numpy as np
from sklearn.cluster import KMeans

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

def retrieve_data(categories = categories):
    dataset = fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 42, categories = categories)
    return dataset

def get_TFIDF(data):
    stop_words = text.ENGLISH_STOP_WORDS
    min_df = 3
    contents = data.data
    vectorizer = TfidfVectorizer(min_df = min_df, stop_words = stop_words)
    tfidf = vectorizer.fit_transform(contents)
    return tfidf

def k_means_cluster(tfidf, k = 2):
    km = KMeans(n_clusters = k, n_init = 100, max_iter = 1000)
    km.fit(tfidf)
    return km

def get_class(data):
    return map(lambda x: int (x < 4), data.target)

def get_result(km, labels):
    homo_score = metrics.homogeneity_score(labels, km.labels_)
    complete_score = metrics.completeness_score(labels, km.labels_)
    v_score = metrics.v_measure_score(labels, km.labels_)
    rand_score = metrics.adjusted_rand_score(labels, km.labels_)
    mutual_info = metrics.adjusted_mutual_info_score(labels, km.labels_)
    return homo_score, complete_score, v_score, rand_score, mutual_info

def print_result(result):
    print("Mogeneity: %0.3f" % result[0]) 
    print("Completeness: %0.3f" % result[1]) 
    print("V-measure: %0.3f" % result[2])
    print("Adjusted rand score: %0.3f" % result[3])
    print("Adjusted mutual info score: %0.3f\n" % result[4])

if __name__ == "__main__":
    dataset = retrieve_data()
    tfidf = get_TFIDF(dataset)
    print tfidf.shape
    km = k_means_cluster(tfidf)
    labels = get_class(dataset)
    result = get_result(km, labels)
    print_result(result)

