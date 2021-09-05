from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import os
from nltk import RegexpTokenizer, word_tokenize, corpus, pos_tag
from nltk.corpus import wordnet
import nltk
#nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
#from transformers import BertTokenizer, TFBertForSequenceClassification
#from transformers import InputExample, InputFeatures

class Sentiment:
    def __init__(self, lemmatization:bool = True):
        self.lemmatization = lemmatization
        self.stemmer = PorterStemmer()
        self.lem = WordNetLemmatizer()
        self.texts_name = ["bible", "upanishads", "quran", "tao", "sumerian", "vedanta"]
        self.stop_words = corpus.stopwords.words('english')
        new_stopwords = ["thee", "ye"]
        self.stop_words.extend(new_stopwords) 
        self.current_dir =  os.path.dirname( os.path.abspath(__file__) )

    def analize_texts(self):
        text_dict = self.open_data()
        text_dict = self.dataprep(text_dict)
        self.lexicon_classifier(text_dict)
    def open_data(self):
        
        text_dict = {}
        for text_name in self.texts_name:
            filename = "{}/data/{}.txt".format(self.current_dir, text_name)
            with open(filename, "r") as f:
                text = f.read()
            text_dict.update({text_name:text})
        return text_dict

    def dataprep(self, text_dict:dict):
        stemmed_dict = {}
        for k, text in text_dict.items():
            tokenizer = RegexpTokenizer(r'\w+')

            words_tokens = tokenizer.tokenize(text)
            #words_tokens = word_tokenize(text)
            words_tokens = [word.lower() for word in words_tokens if word not in self.stop_words]
            words_tokens = [word for word in words_tokens if len(word) > 3]
            words_tokens = [word for word in words_tokens if "gutenberg" not in word]

            wc = WordCloud()
            img = wc.generate_from_text(' '.join(words_tokens))
            img.to_file("{}/images/{}_worcloud.jpeg".format(self.current_dir, k)) 
            
            #pos tagging for further lemmatization
            if self.lemmatization:
                pos_tagged_tokens = pos_tag(words_tokens)
                words_stems = [self.lem.lemmatize(word, self.get_wordnet_pos(tag)) 
                        for word, tag in pos_tagged_tokens if self.get_wordnet_pos(tag)]
                missing = 100*(len(words_tokens)-len(words_stems))/len(words_tokens)
                print("Words not lemmatized : {}%".format(missing))
            else : 
                words_stems = [self.stemmer.stem(word) for word in words_tokens]
            stemmed_dict.update({k:words_stems})
        return stemmed_dict

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lexicon_classifier(self, stem_dict):
        polarity_plot = {}
        subj_plot = {}
        for k, stems in stem_dict.items():
            result = TextBlob(" ".join(stems)).sentiment.polarity
            polarity_plot.update({k:result})
            self.plot_evolution(k, stems)
        
        plt.bar(range(len(polarity_plot)), list(polarity_plot.values()), align='center')
        plt.xticks(range(len(polarity_plot)), list(polarity_plot.keys()))
        plt.savefig("{}/images/polarity_lexicon.jpeg".format(self.current_dir))

    def plot_evolution(self, k, stems):
        list_polarity = [TextBlob(word).sentiment.polarity for word in stems]

        #Let's make a moving average so it's less of a mess
        window_size = 300
        windows = pd.Series(list_polarity).rolling(window_size)
        moving_averages = windows.mean()

        moving_averages_list = moving_averages.tolist()
        list_polarity = moving_averages_list[window_size - 1:]
        size = range(0, len(list_polarity))
        
        fig1, ax1 = plt.subplots()
        ax1.plot(size, list_polarity)
        fig1.savefig("{}/images/{}_evolution.jpeg".format(self.current_dir, k))
        plt.close(fig1)
        








if __name__ == "__main__":
    sa = Sentiment()
    sa.analize_texts()
