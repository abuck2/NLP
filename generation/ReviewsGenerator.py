import pandas as pd
import pickle

# keras module for building LSTM 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
from keras.utils.np_utils import to_categorical

#Data from
#https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews



import numpy as np
import string, os

#Avoid tensorflow warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

class ReviewsGenerator:
    
    def __init__(self, max_corpus_size:int = 100):
        self.current_dir =  os.path.dirname(os.path.abspath(__file__))
        self.tokenizer = Tokenizer()
        self.max_corpus_size = max_corpus_size
    
    def run(self, first_word:str = "This", review_length:int = 20, n_reviews:int = 5, train_model:bool = True, n_epochs:int = 100):
        data = pd.read_csv(self.current_dir+"/data/tripadvisor_hotel_reviews.csv")
        corpus = list(data.Review)[0:self.max_corpus_size]
        features, label, max_sequence_len, total_words = self.dataprep(corpus)
        if train_model :
            model = self.model_builder(max_sequence_len, total_words)
            model.fit(features, label, epochs=n_epochs, verbose=5)
            #pickle.dump(model, open("model.pkl", "wb"))
        else :
            pass
            #model = pickle.load(open("model.p", "rb"))
        generated = []
        for i in range(n_reviews):
            generated.append(first_word)
            first_word = self.generate_text(first_word, review_length, model, max_sequence_len)
        print(" ".join(generated))

    def dataprep(self, corpus):

        #Get tokens and index
        input_sequences = []
        self.tokenizer.fit_on_texts(corpus)
        total_words = len(self.tokenizer.word_index) + 1
        for review in corpus:
            token_list = self.tokenizer.texts_to_sequences([review])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        
        #get same length sentences
        max_sequence_len = max([len(x) for x in input_sequences]) #length of longest sentence
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        
        #Use "next word" as label and previous words as feature
        predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
        label = to_categorical(label, num_classes=total_words)
        return predictors, label, max_sequence_len, total_words

    def model_builder(self, max_sequence_len, total_words):
        input_len = max_sequence_len - 1
        model = Sequential()

        # Add Input Embedding Layer
        model.add(Embedding(total_words, 10, input_length=input_len))

        # Add Hidden Layer 1 - LSTM Layer
        model.add(LSTM(100))
        model.add(Dropout(0.1))

        # Add Output Layer
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return model 
        
    

    def generate_text(self, seed_text:str, next_words:int, model, max_sequence_len):
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predict_x = model.predict(token_list, verbose=0)
            predicted = np.argmax(predict_x,axis=1)

            output_word = ""
            for word,index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " "+output_word
        return seed_text.title()



if __name__=="__main__":
    rg = ReviewsGenerator(max_corpus_size = 500)
    rg.run()
