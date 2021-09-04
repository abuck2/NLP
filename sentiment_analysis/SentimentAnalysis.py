#from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import os
from nltk import RegexpTokenizer, word_tokenize, corpus, pos_tag
import nltk
#nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class Sentiment:
    def __init__(self):
        self.texts_name = ["bible", "upanishads", "quran", "tao", "sumerian", "vedanta"]
        self.stop_words = corpus.stopwords.words('english')
        new_stopwords = ["thee", "ye"]
        self.stop_words.extend(new_stopwords) 
        self.current_dir =  os.path.dirname( os.path.abspath(__file__) )
        #self.stemmer = PorterStemmer()
        self.lem = WordNetLemmatizer()

    def analize_texts(self):
        text_dict = self.open_data()
        text_dict = self.dataprep(text_dict)

    def open_data(self):
        
        text_dict = {}
        for text_name in self.texts_name:
            filename = "{}/data/{}.txt".format(self.current_dir, text_name)
            with open(filename, "r") as f:
                text = f.read()
            text_dict.update({text_name:text})
        return text_dict

    def dataprep(self, text_dict:dict):
        for k, text in text_dict.items():
            tokenizer = RegexpTokenizer(r'\w+')

            words_tokens = tokenizer.tokenize(text)
            #words_tokens = word_tokenize(text)
            words_tokens = [word.lower() for word in words_tokens if word not in self.stop_words]
            words_tokens = [word for word in words_tokens if len(word) > 3]
            words_tokens = [word for word in words_tokens if "gutenberg" not in word]

            wc = WordCloud()
            img = wc.generate_from_text(' '.join(words_tokens))
            img.to_file("{}/images/{}_worcloud.jpeg".format(self.current_dir, k)) # example of something you can do with the img
            
            #pos tagging for further lemmatization
            #words_stems = [self.stemmer.stem(word) for word in words_tokens]
            pos_tagged_tokens = pos_tag(words_tokens)
            #pos_tagged_tokens = [pos_tag(word) for word in words_tokens]
            words_stems = [self.lem.lemmatize(word, self.get_wordnet_pos(tag)) for word, tag in pos_tagged_tokens]


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










if __name__ == "__main__":
    sa = Sentiment()
    sa.analize_texts()
