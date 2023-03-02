from transformers import pipeline
import pandas as pd
from collections import Counter
from heapq import nlargest
import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.lang.en.stop_words import STOP_WORDS
from itertools import islice
import requests
from urllib.parse import unquote
import re
from lxml import html

#preprocessing of the strings
'''
  Input: str
  Output: str
  Preprocessing: Input_String --> Tokenization --> Removal of Stop words and punctuation --> Lemmatization --> Joining of words --> String

'''
def extract_text_from_url(url_title):
    # url = unquote(url)
    # print(url)
    # l=list(url.split('/'))
    title=" ".join(list(url_title.split('_')))
    response = requests.get('https://en.wikipedia.org/w/api.php',
                            params={
                                'action': 'parse',
                                'page': title,
                                'format': 'json',
                            }
                           ).json()
    raw_html = response['parse']['text']['*']
    document = html.document_fromstring(raw_html)
    list_p_tags=list(document.xpath('//p'))
    final_str_list=list()
    for para in list_p_tags:
        text = str(para.text_content())
        final_str_list.append(text)

    final_str="\n".join(final_str_list)
    # print("final_str",final_str)
    final_str=re.sub("[\(\[].*?[\)\]]", "", final_str)
    return final_str
    

class preprocessData:  
    
    def __init__(self,str_input:str):
        
        self.input_para=str_input.lower()
        self.final_str=str()

    def preprocess_string(self):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self.input_para)
        All_punct = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        # st=" ".join([e.text for e in doc.ents if e.label_ not in ("DATE ", "TIME", "MONEY", "ORDINAL","PERCENT","QUANTITY","CARDINAL")])
        # print("st : ",st)
        token_list=list(doc)
        # print(token_list)
        filtered_sentence = [] 

        for word in token_list:
            if word.is_stop == False and str(word) not in All_punct:
                filtered_sentence.append(str(word.lemma_)) 
        self.final_str=" ".join(filtered_sentence)
        self.final_str=self.final_str.replace("\n","")

        
#########################################################
# Defining the summarizer
#########################################################
class summarizer:
    def __init__(self,input_txt):
        self.input_txt=input_txt
        self.freq_keywords=None
        self.tokens_input=str()
        self.sent_strength=dict()
        self.sent_index={}
        self.summary=None
        self.final_summary=str()
        
        
    def calcFrequency(self):
        obj=preprocessData(self.input_txt)
        obj.preprocess_string()
        self.tokens_input=obj.final_str
        tokens_of_article=list(obj.final_str.split(" "))
        tokens_of_article=[x for x in tokens_of_article if x]
        freq_keywords=Counter(tokens_of_article)
        #Normalizing the frequency
        max_freq=freq_keywords.most_common(1)[0][1]
        for word in freq_keywords:
            freq_keywords[word]=(freq_keywords[word]/max_freq)
        return freq_keywords
  
    def sentence_ranking(self):
        nlp = spacy.load("en_core_web_sm")
        doc=nlp(self.input_txt)
        sent_strength={}
        sent_index={}
        i=0
        for sent in doc.sents:
            
            sent_index[sent]=i
            i+=1
            for word in sent:
                if word.text in self.freq_keywords.keys():
                    if sent in sent_strength:
                        sent_strength[sent]+=self.freq_keywords[word.text]
                    else:
                        sent_strength[sent]=self.freq_keywords[word.text]
        return sent_strength, sent_index

    def get_summary(self):
        summary=nlargest(8,self.sent_strength,key=self.sent_strength.get)
        self.summary=summary
        final_dict={}
        for sent in summary:
            final_dict[sent]=self.sent_index[sent]
        final_dict=dict(sorted(final_dict.items(),key=lambda x:x[1]))
        final_sents=[str(sen.text) for sen in list(final_dict.keys())]
        final_summary=" ".join(final_sents)
        final_summary=final_summary.replace("\n",'')
        return final_summary

    
    def wrapper(self):
        self.freq_keywords=self.calcFrequency()
        self.sent_strength,self.sent_index=self.sentence_ranking()
        self.final_summary=self.get_summary()


#########################################################
# Seaching the query
#########################################################
class compute_similarity:
    def __init__(self,input_query,union_df,token_df):
        self.query_input=input_query
        self.union_df=union_df
        self.token_df=token_df
        self.query_tokens_str=str()
        self.assigned_grp=None
        self.matched_ids=None
  
    def preprocess_query(self):
        query_obj=preprocessData(self.query_input)
        query_obj.preprocess_string()
        self.query_tokens_str=query_obj.final_str
  
    def take(self,n, iterable):
        """Return the first n items of the iterable as a list."""
        return list(islice(iterable, n))

  
    def identifying_grp(self):
        self.preprocess_query()
        sim_grp={}
        nlp = spacy.load("en_core_web_sm")
        query_doc=nlp(self.query_tokens_str)
        for i in range(len(self.union_df)):
            union_tokens=self.union_df["union"][i]
            # print(str(self.union_df["intersections"][i]))
            intersect_tokens=str(self.union_df["intersections"][i])
            union_doc = nlp(union_tokens)

            intersect_doc = nlp(intersect_tokens)

            #union score
            union_sim_score=query_doc.similarity(union_doc)

            #intersection score
            intersect_sim_score=query_doc.similarity(intersect_doc)
            final_score=0.6*intersect_sim_score + 0.4*union_sim_score
            sim_grp[self.union_df["group"][i]]=final_score
            
        sorted_score_dict=dict(sorted(sim_grp.items(), key= lambda x:x[1], reverse=True)) 
        self.assigned_grp=list(sorted_score_dict.keys())[0]

    def get_top_n_articles(self):
        n=5
        self.identifying_grp()
        ids=list(map(int, self.union_df["ids"][self.assigned_grp].strip().split(' ')))
        doc_sim={}
        nlp = spacy.load("en_core_web_sm")
        query_doc=nlp(self.query_tokens_str)
        for id in ids:
            ind=self.token_df['id'].loc[lambda x: x==id].index.tolist()
            txt=self.token_df.loc[ind[0],"Tokens"]
            txt_doc=nlp(txt)
            score=txt_doc.similarity(query_doc)
            doc_sim[id]=score
        sorted_score_dict_doc=dict(sorted(doc_sim.items(), key= lambda x:x[1], reverse=True)) 
        self.matched_ids=dict(self.take(n, sorted_score_dict_doc.items()))
  
#########################################################
# Getting Firenliness score
#########################################################

def query(payload):
    API_TOKEN="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"
    API_URL = "https://api-inference.huggingface.co/models/Hate-speech-CNERG/dehatebert-mono-english"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def friendliness_score(query_str:str):
    
    output = query({
        "inputs": query_str,
    })
    return list(output)[0]