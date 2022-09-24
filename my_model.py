# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 04:03:06 2022

@author: kenza
"""

# Forcer l'utilisation de CPU au lieu de GPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

###################################################################################
####################  Importation des bibliothèques à utiliser ####################
###################################################################################

import joblib
from my_functions import *
from sklearn.pipeline import Pipeline


#####################################################################
####################    NETTOYAGE DES DONNEES    ####################
##########    Stopwords, lematization, minuscule, . . .    ##########
#####################################################################

def my_processing ( X ) : 
	my_pipeline = Pipeline([("LowerCase", ToLowerCase() ),
                            ("Ponctuation", Ponctuation() ),
                            ("Stemmatization",Stemmatization() ),
                            ("StopWords",StopWord() ),
                            ("RemoveSpace",RemoveSpace() ) ,
                            ("RemoveURL",RemoveURL() )
                            ])
	
	return my_pipeline.transform( X )


def my_embedding ( X ): 
	encoding_type = "TFIDF"  # Type d'encodage à utiliser
	n_gram_range = (1,1)
	vocab = joblib.load( "./model/vocabulary.joblib" )  # Vocabulaire de mots qui va déterminer  la position des coéfficients de chaque vecteur d'embedding
	
	return document_encoding_algo( X , encoding_type , min_df = 0.0001 , max_df = 1.0, n_gram=n_gram_range, vocabulaire=vocab )

def decode(X) :
	if X == 1: 
		return "Sentiment <strong> Positif </strong>"
	elif X == 0 :
		return "Sentiment <strong> Negatif </strong>"

##########################################################
####################    PREDICTION    ####################
##########################################################

def predict ( text ) :
	# Importation du modèle pré-enrégistré
	model = joblib.load( "./model/modele_simple_XGBOOST.joblib" )
	
	# Traitement du texte
	mon_texte_nettoye = my_processing(text)
	
	# Vectorelisation de texte
	mon_texte_vectorise = my_embedding ( mon_texte_nettoye )
	
	# Prédiction du sentiment dégagé par le texte
	prediction_brute = model.predict( mon_texte_vectorise )[0]
	
	return f"La phrase : `{text}` renvoi theoriquement un " + decode(  prediction_brute  )
	

	
