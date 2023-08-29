#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:05:06 2023

@author: ac
"""

from textblob import TextBlob

# VARIABLES ANALYSÉES #
###############################
APP_VAR_VBT     = 'Raisons_note_recommandation'     # verbatim #

# ANALYSE DE SENTIMENTS #
#########################
# UTILISATION DE TEXTBLOB POUR DÉTERMINER LA POLARITÉ ET LA SUBJECTIVITÉ DES VERBATIMS #
comments['textblob_polarite']       = comments['var_lemma_nosw_txt'].apply(lambda text : TextBlob(text).sentiment.polarity)
comments['textblob_subjectivite']   = comments['var_lemma_nosw_txt'].apply(lambda text : TextBlob(text).sentiment.subjectivity)

# Utilisation du lexique FEEL #
lexique_feel = pd.read_csv("FEEL.csv", sep=',', encoding='utf-8')
liste_positif = lexique_feel.loc[lexique_feel['polarity']=='positive','word'].tolist()
liste_negatif = lexique_feel.loc[lexique_feel['polarity']=='negative','word'].tolist()
liste_joie = lexique_feel.loc[lexique_feel['joy']==1,'word'].tolist()
liste_peur = lexique_feel.loc[lexique_feel['fear']==1,'word'].tolist()
liste_tristesse = lexique_feel.loc[lexique_feel['sadness']==1,'word'].tolist()
liste_colere = lexique_feel.loc[lexique_feel['anger']==1,'word'].tolist()
liste_surprise = lexique_feel.loc[lexique_feel['surprise']==1,'word'].tolist()
liste_degout = lexique_feel.loc[lexique_feel['disgust']==1,'word'].tolist()

comments['nombre_mots_positifs_FEEL']   = comments['var_lemma_nosw'].apply(lambda x : len(set(x) & set(liste_positif)))
#comments['liste_mots_positifs_FEEL']    = comments['var_lemma_nosw'].apply(lambda x : (set(x) & set(liste_positif)))
#comments['pct_mots_positifs_FEEL']      = comments['var_lemma_nosw'].apply(lambda x : (len(set(x) & set(liste_positif)))/max(len(x),1))

comments['nombre_mots_negatifs_FEEL']   = comments['var_lemma_nosw'].apply(lambda x : len(set(x) & set(liste_negatif)))
#comments['liste_mots_negatifs_FEEL']    = comments['var_lemma_nosw'].apply(lambda x : (set(x) & set(liste_negatif)))
#comments['pct_mots_negatifs_FEEL']      = comments['var_lemma_nosw'].apply(lambda x : (len(set(x) & set(liste_negatif)))/max(len(x),1))

comments['nombre_mots_joie_FEEL']       = comments['var_lemma_nosw'].apply(lambda x : len(set(x) & set(liste_joie)))
#comments['pct_mots_joie_FEEL']          = comments['var_lemma_nosw'].apply(lambda x : (len(set(x) & set(liste_joie)))/max(len(x),1))
#comments['liste_mots_joie_FEEL']        = comments['var_lemma_nosw'].apply(lambda x : (set(x) & set(liste_joie)))

comments['nombre_mots_peur_FEEL']       = comments['var_lemma_nosw'].apply(lambda x : len(set(x) & set(liste_peur)))
#comments['pct_mots_peur_FEEL']          = comments['var_lemma_nosw'].apply(lambda x : (len(set(x) & set(liste_peur)))/max(len(x),1))
#comments['liste_mots_peur_FEEL']        = comments['var_lemma_nosw'].apply(lambda x : (set(x) & set(liste_peur)))#

comments['nombre_mots_tristesse_FEEL']  = comments['var_lemma_nosw'].apply(lambda x : len(set(x) & set(liste_tristesse)))
#comments['pct_mots_tristesse_FEEL']     = comments['var_lemma_nosw'].apply(lambda x : (len(set(x) & set(liste_tristesse)))/max(len(x),1))
#comments['liste_mots_tristesse_FEEL']   = comments['var_lemma_nosw'].apply(lambda x : (set(x) & set(liste_tristesse)))

comments['nombre_mots_colere_FEEL']     = comments['var_lemma_nosw'].apply(lambda x : len(set(x) & set(liste_colere)))
#comments['pct_mots_colere_FEEL']        = comments['var_lemma_nosw'].apply(lambda x : (len(set(x) & set(liste_colere)))/max(len(x),1))
#comments['liste_mots_colere_FEEL']      = comments['var_lemma_nosw'].apply(lambda x : (set(x) & set(liste_colere)))

comments['nombre_mots_surprise_FEEL']   = comments['var_lemma_nosw'].apply(lambda x : len(set(x) & set(liste_surprise)))
#comments['pct_mots_surprise_FEEL']      = comments['var_lemma_nosw'].apply(lambda x : (len(set(x) & set(liste_surprise)))/max(len(x),1))
#comments['liste_mots_surprise_FEEL']    = comments['var_lemma_nosw'].apply(lambda x : (set(x) & set(liste_surprise)))

comments['nombre_mots_degout_FEEL']     = comments['var_lemma_nosw'].apply(lambda x : len(set(x) & set(liste_degout)))
#comments['pct_mots_degout_FEEL']        = comments['var_lemma_nosw'].apply(lambda x : (len(set(x) & set(liste_degout)))/max(len(x),1))
#comments['liste_mots_degout_FEEL']      = comments['var_lemma_nosw'].apply(lambda x : (set(x) & set(liste_degout)))







