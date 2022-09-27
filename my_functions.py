# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 04:03:06 2022

@author: kenza
"""
#####################################################
# Forcer l'utilisation de CPU au lieu de GPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#####################################################
import nltk
import spacy
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer

#####################################################
#en_core_web_md
#en_core_web_sm
try:
    nlp = spacy.load("en_core_web_md")
except: # If not present, we download
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

#####################################################
class ToLowerCase(BaseEstimator ,TransformerMixin ): 
    def fit ( self , X , y = None ): 
        return self
    
    def transform ( self , X) :
        return  X.lower() 


class Lemmatization(BaseEstimator ,TransformerMixin ): 
    def fit ( self , X , y = None ): 
        return self
    
    def transform ( self , X) :
        return " ".join( token.lemma_ for token in nlp(X))

class Stemmatization(BaseEstimator ,TransformerMixin ): 
    def fit ( self , X , y = None ): 
        return self
    
    def transform ( self , X) :
        stemmer = nltk.PorterStemmer()
        return " ".join( stemmer.stem(token.text) for token in nlp(X))

class StopWord(BaseEstimator ,TransformerMixin ): 
    def fit ( self , X , y = None ): 
        return self
    
    def transform ( self , X) :
        stpwd = [":|",":","|","ã","½","¿","ƒ","iãƒâ¯ã‚â¿ã‚â½in","¯","canãƒâ¯ã‚â¿ã‚â½t","=/",":-p",":p","-p","/","=",":|","ãƒâ¯ã‚â¿ã‚â½i",
                 "thãƒâ¯ã‚â¿ã‚â½n","khãƒâ¯ã‚â¿ã‚â½m","ãƒâ¯ã‚â¿ã‚â½","â"]
        return " ".join( token.text for token in nlp(X) if not token.is_stop and token.text not in stpwd )
    
class Ponctuation(BaseEstimator ,TransformerMixin ): 
    def fit ( self , X , y = None ): 
        return self
    
    def transform ( self , X) :
        return " ".join( token.text for token in nlp(X) if not token.is_punct )
    
class RemoveSpace(BaseEstimator ,TransformerMixin ): 
    def fit ( self , X , y = None ): 
        return self
    
    def transform ( self , X) :
        return " ".join( token.text for token in nlp(X) if not token.is_space ) 
    
class RemoveURL(BaseEstimator ,TransformerMixin ): 
    def fit ( self , X , y = None ): 
        return self
    
    def transform ( self , X) :
        return " ".join(token.text for token in nlp(X) if not token.like_url )

#####################################################

def document_encoding_algo ( X , model_encoding = "USE",n_gram=(1,1), min_df = 0.001 , max_df = 1. , vocabulaire=None, get_vocabulary=False) :
    col = "text"
    data = pd.DataFrame( {col : X } , index = [0] )
    
    if model_encoding == "USE" :
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
        model_use = hub.load(module_url)
        
        return pd.DataFrame.from_dict( {  f"{k}": np.array(model_use( [data[col][k]] )).reshape(512,) for k in data.index  } , orient='index' )
    elif model_encoding == "TFIDF" :
        
        tfidf_vect = TfidfVectorizer( ngram_range=n_gram, min_df = min_df , max_df = max_df, vocabulary = vocabulaire )
        tfIdf = tfidf_vect.fit_transform(data[col]).toarray()
        
        if not get_vocabulary :
            return pd.DataFrame(tfIdf, columns= tfidf_vect.get_feature_names())
        else :
            return tfidf_vect.vocabulary_

    elif model_encoding == "countvectorizer" :
        count_vect = CountVectorizer(ngram_range=n_gram, min_df = min_df , max_df = max_df, vocabulary = vocabulaire )
        out = count_vect.fit_transform(data[col]).toarray()
        
        if not get_vocabulary :
            return pd.DataFrame( out , columns=count_vect.get_feature_names())
        else :
            return count_vect.vocabulary_
    else :
        raise ValueError(f"Le modèle d'encodage \"{model_encoding}\" n'est pas disponible")
        
        

