# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 04:03:34 2022

@author: kenza
"""

import numpy as np
import my_model
from flask import Flask, jsonify, request, render_template
 

app = Flask( __name__ )
 
@app.route("/", methods=['GET'])
def home():
    return "<h1> API de prédiction de sentiment </h1> <p> <h4>Developpée par Brice KENGNI ZANGUIM</h4> </p> <p> Bien vouloir entrer le texte dans la barre d'URL en \
    utilisant l'URL   <strong>/predict?text=`Votre texte ici` </strong> pour faire des prédictions</p> <p> A noter que le caractère <strong>ESPACE</strong> est à remplacer par <strong>%20</strong> </p>"
    
    
@app.route("/predict", methods=['GET'])
def get_prediction():
    text = request.args.get('text')
    polarity = my_model.predict(text)
 
    return jsonify(polarity)
################################################################################
################################################################################

if __name__ == "__main__":
    
    app.run(debug=False)
