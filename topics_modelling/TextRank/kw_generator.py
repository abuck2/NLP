import pandas as pd
import numpy as np
import time, sys, os
from pprint import pprint
sys.path.append("../../KeywordsDetection")
sys.path.append("../../Utils")
from KeywordExtractor_TextRazor import KeywordsExtractor_TextRazor

#Open the data
data=pd.read_csv('competitors_data.csv')



keywords_extractor = KeywordsExtractor_TextRazor("BE")

keywords=[]
for index, item in enumerate(data.website):
    pprint(item)
    if isinstance(item, float):
        keywords.append(item)
    else:
        if item == 'https://www.facebook.com/Oulaoops-1817729788471637/?fref=ts':
            kw=None
        else :
            kw=keywords_extractor.get_keywords_from_url(item)
        keywords.append(kw)

data['Keywords']=keywords
data.to_csv('competitors_data_kw.csv')

























