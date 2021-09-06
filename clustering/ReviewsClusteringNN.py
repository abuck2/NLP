#Models, embedding and evaluation
from gensim.utils import simple_preprocess, tokenize
from gensim.models import doc2vec
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


#Wordcloud
from wordcloud import WordCloud


import pandas as pd 
import numpy as np
import os



class ReviewsClustering:
    def __init__(self, model_type = "kmeans"):
        print("loading models") 
        self.current_dir =  os.path.dirname(os.path.abspath(__file__))        
        self.model_type = model_type
        if model_type == "DBSCAN":
            self.model = DBSCAN(eps=0.9)
        elif model_type == "kmeans":
            self.model = MiniBatchKMeans(n_clusters=25, random_state=0, batch_size=100, max_iter=20)
        elif model_type == "hierarchical":
            self.model = AgglomerativeClustering(n_clusters = 5, linkage = "complete", affinity = "l1")
        else : 
            raise ValueError("model not supported")

        self.embedder = doc2vec.Doc2Vec(vector_size=500, min_count=2, epochs=100)

    def run(self, train:bool = True, vectorize = True):
        data = pd.read_csv(self.current_dir+"/data/tripadvisor_hotel_reviews.csv")
        data = self.dataprep(data, train, vectorize)
        clusters = self.cluster_data(data)
        self.evaluation_perf(clusters)

    def dataprep(self, data, train, vectorize):
        print("Tagging")
        list_docs = [doc2vec.TaggedDocument(words=simple_preprocess(row["Review"]), tags=[index]) for index, row in data.iterrows()]
        if train : 
            print("Building embedder")
            self.embedder.build_vocab(list_docs)
            self.embedder.train(list_docs, total_examples=self.embedder.corpus_count, epochs=100) 
            self.embedder.save(self.current_dir+"/d2v.mod")
        else : 
            self.embedder = doc2vec.Doc2Vec.load(self.current_dir+"/d2v.mod")
        
        if vectorize:
            print("Creating vectors")
            vectors = []
            index_list = [] 
            for index, review in enumerate(data.Review):
                words = simple_preprocess(review) #tokenize, lowercase, deaccent
                vector = self.embedder.infer_vector(words)
                vectors.append(vector)
            data = pd.DataFrame(vectors, index = data.Review)
            print(data.shape)
            data.to_csv(self.current_dir+"/data/embeddings.csv")
        else : 
            data = pd.read_csv(self.current_dir+"/data/embeddings.csv")
            print(data.shape)
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
    rc.run(train = True, vectorize = True)
