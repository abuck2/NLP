#import urllib.request
#from urllib.parse import  urljoin
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, SoupStrainer
import os


#Data from https://www.kaggle.com/tentotheminus9/religious-and-philosophical-texts

class TextScraper:
    def __init__(self):
        self.current_dir =  os.path.dirname( os.path.abspath(__file__) ) 
        print(self.current_dir)

    def get_data(self):

        #urls
        bible_url = "https://www.gutenberg.org/files/10/10-h/10-h.htm"
        upanishads = "https://www.gutenberg.org/cache/epub/3283/pg3283.html"
        quran = "https://www.gutenberg.org/cache/epub/3434/pg3434.html"
        tao = "https://www.gutenberg.org/files/216/216-h/216-h.htm"
        sumerian = "https://www.gutenberg.org/files/31935/31935-h/31935-h.html"
        vedanta = "https://www.gutenberg.org/files/16295/16295-h/16295-h.htm"
        dict_urls = {"bible":bible_url, "upanishads" :upanishads, "quran":quran, "tao":tao, "sumerian":sumerian, "vedanta":vedanta} 

        dict_content = {}
        for key, url in dict_urls.items() :
            print('scraping {}...'.format(key))
            text = self.get_page_content(url)
            filename = "{}/data/{}.txt".format(self.current_dir, key) 
            with open(filename, 'w') as f:
                f.write(text)
            dict_content.update({key:text})
        return dict_content
    
    def get_links_test(self, url):
        http = httplib2.Http()
        status, response = http.request(url)
        links_list = []
        for link in BeautifulSoup(response, 'html.parser', parse_only=SoupStrainer('a')):
            if link.has_attr('href'):
                links_list.append(urljoin(url, link['href']))
        return links_list

    def get_page_content(self, url):
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        text = soup.get_text()
        return text

if __name__=="__main__":
    fb = TextScraper()
    fb.get_data()
