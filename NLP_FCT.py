#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:41:39 2020

@author: Anne-Cécile GAY
"""

###############################################################################
#                               FONCTIONS                                     #
###############################################################################                       

from collections import Counter
import numpy as np
import pandas as pd
import fr_core_news_sm
nlp = fr_core_news_sm.load()
from gensim.utils import simple_preprocess
from nltk.util import ngrams
from sklearn.manifold import TSNE


# TOKENISATION #
################
def return_token_gensim(sentence) :
    return simple_preprocess(sentence, deacc=True, min_len=2)

# SUPPRESSION DES STOPWORDS #
#############################
def supp_sw(sentence_list,sw_list) :
    return [w for w in sentence_list if not w in sw_list]

# LEMMATISATION #
#################
def return_lemma(sentence):
    doc = nlp(sentence)
    return_lemma = ' '.join([X.lemma_ for X in doc])
    return(return_lemma)
       
#  VOCABULAIRE DU CORPUS  #
###########################
def vocab(df, var_to_vocab, freq_min) :
    print('*****     CREATION DU DF DE VOCABULAIRE     *****')
    list_vocabulary = Counter([x for elem in df[var_to_vocab].tolist() for x in elem])
    df_vocabulary = pd.DataFrame.from_dict(list_vocabulary,orient='index').reset_index()
    df_vocabulary.rename(columns={'index':'word', 0:'effectif'}, inplace=True)
    df_vocabulary.sort_values(by = ['effectif','word'], ascending = False,inplace=True)
    df_vocabulary.reset_index(drop=True,inplace=True)
    df_vocabulary = df_vocabulary.loc[df_vocabulary['effectif']>freq_min]
    return(df_vocabulary)
    
# EXTRACTION DES N-GRAMS D'UN TEXTE # 
#####################################    
def extract_ngrams(sentence, num):
    n_grams = ngrams(sentence, num)
    return [ ' '.join(grams) for grams in n_grams]   

#  N-GRAMS DU CORPUS  #
#######################
def vocab_ngrams(df, var_to_vocab, freq_min) : 
    print('*****     CREATION DU DF DE N-GRAMS         *****')
    df['text_ngrams2'] = df[var_to_vocab].apply(lambda text : extract_ngrams(text,2))
    df['text_ngrams3'] = df[var_to_vocab].apply(lambda text : extract_ngrams(text,3))    
    df['text_ngrams']  = df['text_ngrams2'] + df['text_ngrams3']
    
    list_ngrams = Counter([x for elem in df['text_ngrams'].tolist() for x in elem])
    
    df_ngrams = pd.DataFrame.from_dict(list_ngrams,orient='index').reset_index()
    df_ngrams.rename(columns={'index':'ngrams', 0:'effectif'}, inplace=True)
    df_ngrams.sort_values(by = ['effectif','ngrams'], ascending = False,inplace=True)
    df_ngrams.reset_index(drop=True,inplace=True)    
    df_ngrams = df_ngrams.loc[df_ngrams['effectif']>freq_min]
    #df_ngrams = df_ngrams.sort_values(by = 'effectif', ascending = False)
    return (df_ngrams)

# DETECTION DES PoS #
#####################
def return_POS(sentence):
    doc = nlp(sentence)
    return [(X, X.pos_, X.lemma_, X.tag_, X.dep_) for X in doc]    
   
# EXTRACTION DES MOTS D'UN TEXTE EN FONCTION DE LEUR NATURE GRAMMATICALE #
##########################################################################    
def return_naturemot(list_words,naturemot):
    sentence = ' '.join([word for word in list_words])
    return_naturemot = [X.lemma_ for (X, X.pos_, X.lemma_, X.tag_, X.dep_) in return_POS(sentence) if X.pos_==naturemot]
    return [w for w in return_naturemot]
   
# NATURES GRAMMATICALES DU CORPUS #
###################################
def pos(df, var_to_vocab, freq_min) :
    print('*****           DETECTION PoS               *****')   
    df['verb']   = df[var_to_vocab].apply(lambda x: return_naturemot(x,'VERB'))
    df['adj']    = df[var_to_vocab].apply(lambda x: return_naturemot(x,'ADJ'))
    # df['adv']    = df[var_to_vocab].apply(lambda x: return_naturemot(x,'ADV'))
    # df['punct']  = df[var_to_vocab].apply(lambda x: return_naturemot(x,'PUNCT'))
 
    print('*****     CREATION DU DF PoS : VERBES       *****')    
    list_verb = Counter([x for elem in df['verb'].tolist() for x in elem])
    df_verb = pd.DataFrame.from_dict(list_verb,orient='index').reset_index()
    df_verb.rename(columns={'index':'word', 0:'effectif'}, inplace=True)  
    df_verb.sort_values(by = ['effectif','word'], ascending = False,inplace=True)
    df_verb.reset_index(drop=True,inplace=True)  
    df_verb = df_verb.loc[df_verb['effectif']>freq_min]

    print('*****     CREATION DU DF PoS : ADJECTIFS    *****')   
    list_adj = Counter([x for elem in df['adj'].tolist() for x in elem])
    df_adj = pd.DataFrame.from_dict(list_adj,orient='index').reset_index()
    df_adj.rename(columns={'index':'word', 0:'effectif'}, inplace=True)  
    df_adj.sort_values(by = ['effectif','word'], ascending = False,inplace=True)
    df_adj.reset_index(drop=True,inplace=True)  
    df_adj = df_adj.loc[df_adj['effectif']>freq_min]
    return(df_verb, df_adj)


 # PRE-PROCESSING COMPLET #
##########################
def preprocessing(df,APP_VAR,sw_list,freq_min):
    print('******************************************************************')
    print('                 PRE-TRAITEMENT DES DONNÉES                       ')
    print('******************************************************************')
    # Lemmatisation, tokenisation, suppression des stop-words #
    print('*****             LEMMATISATION             *****')
    df['var_lemma']         = df[APP_VAR].apply(lambda text : return_lemma(text))  
    print('*****              TOKENISATION             *****')
    df['var_lemma_token']   = df['var_lemma'].apply(lambda x: return_token_gensim(x))
    print('*****      SUPPRESSION DES STOP-WORDS       *****')
    df['var_lemma_nosw']    = df['var_lemma_token'].apply(lambda text : supp_sw(text,sw_list))   
    
    # Extraction du vocabulaire #
    df_vocabulary = vocab(df, 'var_lemma_nosw', freq_min) 
    df_vocabulary.sort_values(by = 'word', inplace=True)
    
    # Extraction des n-grams #
    #df_ngrams = vocab_ngrams(df, 'var_lemma_nosw', freq_min)  
    
    # Extraction des PoS #
    #df_verb, df_adj = pos(df, 'var_lemma_nosw', freq_min)
    
    #return(df, df_vocabulary, df_ngrams, df_verb, df_adj)
    return(df,df_vocabulary)

# EXTRACTION DES COMMENTAIRES CONTENANT UN MOT DONNÉ #
######################################################
def fct_filtre_word(df,var_to_explore,word):  
    df_filtre_word = df.loc[df[var_to_explore].str.contains(word,regex=False).fillna(False)]  
    return(df_filtre_word)
  
# EXTRACTION DES COMMENTAIRES CONTENANT UN N-GRAM DONNÉ #
#########################################################    
def fct_filtre_ngrams(df,var_to_explore, ngrams):   
    df_filtre_ngrams = df.loc[df[var_to_explore].str.contains(ngrams, regex=False).fillna(False)]
    return(df_filtre_ngrams)
 
# REDUCTION DE DIMENSION APPLIQUÉE A UNE VECTORISATION WORD2VEC #
#################################################################
def reduce_dimensions_tsne(model):
    num_dimensions = 2  # taille de la dimension souhaitée #
    vectors = []
    labels = []
    for word in model.wv.vocab:
        vectors.append(model.wv[word])
        labels.append(word)
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # Réduction de dimension par t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels, vectors    


# COULEURS POUR NUAGES DE MOTS #
################################ 
def random_color_func_pos(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    h = 190
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)


    
    























    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    