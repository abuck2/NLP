import pandas as pd 
import numpy as np
import sys
import spacy
from pprint import pprint
sys.path.append("/home/alexis/inoopadata//Utils")
sys.path.append("/home/alexis/inoopadata/NacePrediction/NewArchitecture")
from WebsiteContentSelector import WebsiteContentSelector
from collections import OrderedDict
from KwSpacy import TextRank4Keyword
from langdetect import detect
## To replace


class KeywordExtractor():
    
    def __init__(self, country):
        self.country=country
        self.content_selector = WebsiteContentSelector(country=self.country)
        self.languages = self.content_selector.languages
        self.tr4w = TextRank4Keyword()
        self.blacklist=["D'", "•","D’"]

    def get_website_content(self):
        
        #pprint(self.content_selector.order_website_content_by_relevance(url=self.website, language=self.languages))
        text=self.content_selector.order_website_content_by_relevance(url=self.website, language=self.language)[0][0]
        string_text=''.join(text)
        
        #POS Tagging
        self.tr4w.analyze(string_text,self.language, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=True,stopwords=self.blacklist)
        self.tr4w.get_keywords(10)



    def get_keywords_from_text():
        pass
    
    def get_keywords(self, website):
        self.website=website
        
        ##### To change later
        self.language='fr'
        self.get_website_content()


if __name__=='__main__':
    extractor=KeywordExtractor('BE')
    data=extractor.get_keywords("https://www.supergrandeboucherie.be/")
    #data=extractor.get_keywords("http://www.chauraci.be")


