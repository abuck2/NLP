import tensorflow_hub as hub
import tensorflow as tf
#import tensorflow_text
import os
#To do : open data, tokenizer, recreate vocab.txt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class ReviewClustering:
    def __init__(self):
        print("initialization")
        print("Tensorflow Version: ", tf.__version__)
        print("Eager mode: ", tf.executing_eagerly())
        print("Hub version: ", hub.__version__)
        device_name = tf.test.gpu_device_name()
        print('Found GPU at: {}'.format(device_name))

        #Import BErt
        BERT_URL = 'https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1'
        self.bert_module = hub.load(BERT_URL)
        #create tokenizer here
        self.tokenizer = Tokenizer()

    def run(self):
        #Open data here
        pd.read_csv("tripadvisor_hotel_reviews.csv")
        text = list(data.Review)

        self.max_seq_len = max([len(elem) for elem in data.Review])
        self.dataprep(text)


    def dataprep(self, text):
        print("Dataprep")
        #Tokens
        input_ids_vals, input_mask_vals, segment_ids_vals  = self.sent_to_features(text)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
        input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None])
        segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])

        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        bert_outputs = self.bert_module(bert_inputs, signature="tokens", as_dict=True)
        
        print("Run vectorizer")
        out = sess.run(bert_outputs, feed_dict={input_ids: input_ids_vals, 
            input_mask: input_mask_vals, 
            segment_ids: segment_ids_vals}
            )

        return out

    def sent_to_features(self, sentences):
        
        self.tokenizer.fit_on_texts(sentences)
        self.vocab = self.get_vocab(sentences)
        
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        
        for index, sentence in enumerate(sentences):
            print("feature {}".format(index))
            input_ids, input_mask, segment_ids = self.unique_sent_to_features(sentence)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)

        return all_input_ids, all_input_mask, all_segment_ids

    def get_vocab(self):
        #Create vocab file
        print(self.tokenizer.word_index)
        list_words = list(self.tokenizer.word_index)
        vocab = {word:index for index, word in enumerate(list_words)}
        return vocab

    def unique_sent_to_features(self, sentence):
        
        #Tokenize sentence with opening and closing token
        tokens = ['[CLS]']
        tokens.extend(self.tokenizer.texts_to_sequences([sentence]))
        if len(tokens) > self.max_seq_len-1:
            tokens = tokens[:self.max_seq_len-1] #Padding/truncating
        tokens.append('[SEP]')

        segment_ids = [0] * len(tokens)
        input_ids = self.convert_tokens_to_ids(tokens) #Check
        input_mask = [1] * len(input_ids)

        #padding with 0
        zero_mask = [0] * (self.max_seq_len-len(tokens))
        input_ids.extend(zero_mask)
        input_mask.extend(zero_mask)
        segment_ids.extend(zero_mask)

        return input_ids, input_mask, segment_ids

    def convert_tokens_to_ids(tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids


if __name__=="__main__":
    rc = ReviewClustering()
    rc.run()
