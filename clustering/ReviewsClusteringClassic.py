#Models, embedding and evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

#Wordcloud
from wordcloud import WordCloud


import pandas as pd 
import numpy as np
import os



class ReviewsClustering:
    def __init__(self, model_type = "kmeans"):
        self.vectorizer = TfidfVectorizer(stop_words={'english'})
        self.current_dir =  os.path.dirname(os.path.abspath(__file__))        
        self.dim_red = TruncatedSVD(500)
        self.model_type = model_type
        if model_type == "DBSCAN":
            self.model = DBSCAN(eps=0.7)
        elif model_type == "kmeans":
            self.model = MiniBatchKMeans(n_clusters=5, random_state=0, batch_size=100, max_iter=20)
        elif model_type == "hierarchical":
            self.model = AgglomerativeClustering(n_clusters = 10, linkage = "ward", affinity = "euclidean")
        else : 
            raise ValueError("model not supported")

    def run(self):
        data = pd.read_csv(self.current_dir+"/data/tripadvisor_hotel_reviews.csv")
        reviews_list = list(data.Review)
        data = self.dataprep(reviews_list)
        clusters = self.cluster_data(data)
        self.evaluation_perf(clusters)

    def dataprep(self, reviews_list):
        X = self.vectorizer.fit_transform(reviews_list)
        print("initial shape : {}".format(X.shape))
        Xpca = self.dim_red.fit_transform(X)
        print("After PCA : {}".format(Xpca.shape))
        explained = self.dim_red.explained_variance_ratio_.sum()
        exp = round(explained*100, 2)
        print("Percentage of explained variance after SVD : {}%".format(exp))
        data = pd.DataFrame(Xpca, index = reviews_list)
        return data

    def cluster_data(self, data):
        self.model.fit(data)
        if self.model_type == "DBSCAN" or self.model_type == 'hierarchical':
            data["cluster"] = self.model.labels_
            n_cluster = len(list(set(data.cluster)))
            print("DBSCAN made {} different clusters".format(n_cluster))
        else : 
            data['cluster'] = self.model.predict(data)
        return data

    def evaluation_perf(self, clusters):
        clusters_set = list(set(clusters.cluster))
        for cluster in clusters_set:
            text_cluster = clusters[clusters.cluster == cluster]
            texts_list = list(text_cluster.index)
            
            STOPWORDS = ["hotel", "room"]
            wc = WordCloud(stopwords=STOPWORDS)
            img = wc.generate_from_text(' '.join(texts_list))
            img.to_file("{}/images/{}_{}_worcloud.jpeg".format(self.current_dir,self.model_type, cluster))
        
        ch_score = calinski_harabasz_score(clusters.drop('cluster', 1), clusters.cluster)
        print("ch score = {}. Higher is better.".format(ch_score))

        db_score = davies_bouldin_score(clusters.drop('cluster', 1), clusters.cluster)
        print("db score = {}. Lower is better.".format(db_score))

        db_score = silhouette_score(clusters.drop('cluster', 1), clusters.cluster)
        print("silhouette score = {}. Higher is better.".format(db_score))

if __name__=="__main__":
    rc = ReviewsClustering(model_type = "hierarchical")
    rc.run()
