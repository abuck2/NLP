#Models, embedding and evaluation
#spacy.require_gpu()
#spacy.prefer_gpu()
from gensim.utils import simple_preprocess, tokenize
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

#Wordcloud
from wordcloud import WordCloud
import torch
from transformers import BertTokenizer, BertModel

import pandas as pd 
import numpy as np
import os, time
import matplotlib.pyplot as plt


class ReviewsClustering:
    def __init__(self, model_type = "kmeans"):
        print("loading models") 
        self.current_dir =  os.path.dirname(os.path.abspath(__file__))        
        self.model_type = model_type
        if model_type == "DBSCAN":
            self.model = DBSCAN(eps=0.9)
        elif model_type == "kmeans":
            self.model = MiniBatchKMeans(n_clusters=5, random_state=0, batch_size=100, max_iter=20)
        elif model_type == "hierarchical":
            self.model = AgglomerativeClustering(n_clusters = 15, linkage = "complete", affinity = "l1")
        else : 
            raise ValueError("model not supported")

        model_name = "en_core_web_trf"
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def run(self, train:bool = True, vectorize = True):
        data = pd.read_csv(self.current_dir+"/data/tripadvisor_hotel_reviews.csv")
        data = self.dataprep(data, train, vectorize)
        clusters = self.cluster_data(data)
        self.evaluation_perf(clusters)

    def dataprep(self, data, train, vectorize):

        print("Tokenize")
        tagged_rev = ["[CLS] " + text + " [SEP]" for text in data.Review]
        tokenized_rev = [self.tokenizer.tokenize(marked_text) for marked_text in tagged_rev]
        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in tokenized_rev]
        segments_ids = [[index] * len(tokenized_text) for index, tokenized_text in enumerate(tokenized_rev)]
        
        tokens_tensor = [torch.tensor([tok]) for tok in indexed_tokens]
        segments_tensors = [torch.tensor([ids]) for ids in segments_ids]

        print("Loading BERT")
        model = BertModel.from_pretrained('bert-base-uncased',
                        output_hidden_states = True)

        print("Evaluation")
        model.eval()
        errors = 0
        with torch.no_grad():
            for idx, toks in enumerate(tokens_tensor):
                print("Processing Review {}".format(idx))
                try:
                    outputs = model(toks, segments_tensors[idx])
                    hidden_states = outputs[2]
                    token_embeddings = torch.stack(hidden_states, dim=0)
                    token_embeddings = torch.squeeze(token_embeddings, dim=1)
                    token_embeddings = token_embeddings.permute(1,0,2)
                    token_vecs = hidden_states[-2][0]
                    sentence_embedding = torch.mean(token_vecs, dim=0)
                    print(sentence_embedding.tolist())
                except:
                    errors += 1
                    print(errors)
        print(errors)
        print(data.shape)
        raise ValueError()


        return data

    def cluster_data(self, data):
        print("clustering")
        self.model.fit(pd.DataFrame(list(data.vector)))
        if self.model_type == "DBSCAN" or self.model_type == 'hierarchical':
            data["cluster"] = self.model.labels_
            n_cluster = len(list(set(data.cluster)))
            print("DBSCAN made {} different clusters".format(n_cluster))
        else : 
            data['cluster'] = self.model.predict(pd.DataFrame(list(data.vector)))
        data.index = data.Review
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
    sart = time.time()
    rc = ReviewsClustering(model_type = "hierarchical")
    rc.run(train = True, vectorize = True)
    print(time-time()-start)
