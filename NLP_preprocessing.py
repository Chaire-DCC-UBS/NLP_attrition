#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 10:50:29 2020

@author: Anne-Cécile GAY
"""

###############################################################################
#                            PREPROCESSING                                    #
###############################################################################

# Chargement des librairies utiles #
####################################
#import APPLI_NLP_FCT
#import os
#import pandas as pd


# Lemmatisation, tokenisation, suppression des stop-words #
##########################################################
#comments, df_vocabulary, df_ngrams, df_verb, df_adj = NLP_FCT.preprocessing(comments,APP_VAR,sw,freq_min_word_to_show)
comments, df_vocabulary = NLP_FCT.preprocessing(comments,APP_VAR,sw,freq_min_word_to_show)
comments['var_lemma_nosw_txt'] = comments['var_lemma_nosw'].apply(lambda mylist : ' '.join(mylist))


