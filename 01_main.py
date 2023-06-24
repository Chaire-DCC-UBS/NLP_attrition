#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:42:04 2020

@author: Anne-Cécile GAY
"""

from collections import Counter, OrderedDict
from gensim.test.utils import datapath
from gensim.models import word2vec, Doc2Vec
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
import numpy as np 
import os
import pandas as pd
import random 
import spacy 
#from sklearn.feature_extraction.text import CountVectorizer 
import matplotlib.pyplot as plt
import NLP_FCT
from wordcloud import WordCloud






###############################################################################
#                               PARAMETRES                                    #
###############################################################################

# CHARGEMENT DES PARAMETRES FICHIERS ET VARIABLES #
###################################################
#param_app_to_load   = pd.read_csv(os.path.join(APP_DIR, 'appli_nlp_descr_parameters.csv'), sep=',', encoding='utf-8')
#param_app_to_load.set_index('parameter_name',inplace=True)

#APP_FILE                            = param_app_to_load.loc['file_name','parameter_value']
#APP_VAR                             = param_app_to_load.loc['var_name','parameter_value']
#freq_min_word_to_show               = int(param_app_to_load.loc['freq_min_word_to_show','parameter_value'])
freq_min_word_to_show                = 0
#min_similitude_word_to_highlight    = float(param_app_to_load.loc['min_similitude_word_to_highlight','parameter_value'])
#APP_model_w2v                       = param_app_to_load.loc['w2vmodel_name','parameter_value']


#  CHARGEMENT DES STOPWORDS ADDITIONNELS USERS  #
#################################################
#sw_app_to_load   = pd.read_csv(os.path.join(APP_DIR, 'appli_nlp_stopwords.csv'), sep=',', encoding='utf-8')
#sw_user = list(sw_app_to_load['stopwords_user'])
sw_user = ['etre', 'avoir', 'cela', 'car']
sw = list(stopwords.words('french'))+sw_user


# NOM DE LA VARIABLE A ANALYSER #
#################################
#APP_VAR = 'Recommandation_amelioration'
#APP_VAR = 'Recommandation_commentaire'
APP_VAR = 'Raisons_note_recommandation'


###############################################################################
#                 CHARGEMENT DES DONNEES ET PRE-TRAITEMENTS                   #
###############################################################################

# CHARGEMENT DES DONNÉES #
#comments = pd.read_csv('data_tests/data_attrition_allianz.csv', sep=';', encoding='utf-8')
#comments = pd.read_excel('data_tests/data_attrition_thelem.xlsx')
#comments = comments_attr_conc_sans.copy()
#comments = comments_attr_conc_part.copy()
#comments = comments_attr_conc_tot.copy()
comments = comments_enquete_client.copy()
comments = comments.dropna(subset=[APP_VAR])

# PREPROCESSING
exec(open('NLP_preprocessing.py').read())







