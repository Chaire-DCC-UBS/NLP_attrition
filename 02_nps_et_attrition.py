#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:38:19 2022

@author: ac
"""

# Bibliothèques #
import collections
import numpy as np
import pandas as pd

##############################################################################
#                            QUELQUES STATS                                  #
##############################################################################

# LIEN ENTRE NOTE NPS ET ATTRITION #
np.mean(comments_enquete_client.loc[comments_enquete_client['ATTR_ALL']=='SANS']['Note_recommandation'])    
np.mean(comments_enquete_client.loc[comments_enquete_client['ATTR_ALL']=='PART']['Note_recommandation'])    
np.mean(comments_enquete_client.loc[comments_enquete_client['ATTR_ALL']=='TOT']['Note_recommandation'])    

np.mean(comments_enquete_client.loc[comments_enquete_client['ATTR_CONC']=='SANS']['Note_recommandation'])    
np.mean(comments_enquete_client.loc[comments_enquete_client['ATTR_CONC']=='PART']['Note_recommandation'])    
np.mean(comments_enquete_client.loc[comments_enquete_client['ATTR_CONC']=='TOT']['Note_recommandation'])   

# CREATION D'UN DATAFRAME : TAUX D'ATTRITION PAR NOTE NPS : en effectif #
# Sans attrition #
df_note_attr_eff = pd.DataFrame(comments_enquete_client.loc[comments_enquete_client['ATTR_CONC']=='SANS']['Note_recommandation'].value_counts(dropna=False))
df_note_attr_eff.rename(columns = {'Note_recommandation': 'tx_attr_sans'}, inplace=True) 

# Taux d'attrition partielle #
temp_note_attr_part_conc = pd.DataFrame(comments_enquete_client.loc[comments_enquete_client['ATTR_CONC']=='PART']['Note_recommandation'].value_counts(dropna=False))
temp_note_attr_part_conc.rename(columns = {'Note_recommandation': 'tx_attr_partielle'}, inplace=True) 
df_note_attr_eff = pd.merge(df_note_attr_eff, temp_note_attr_part_conc, left_index=True, right_index=True)

# Taux d'attrition totale #
temp_note_attr_tot_conc = pd.DataFrame(comments_enquete_client.loc[comments_enquete_client['ATTR_CONC']=='TOT']['Note_recommandation'].value_counts(dropna=False))
temp_note_attr_tot_conc.rename(columns = {'Note_recommandation': 'tx_attr_totale'}, inplace=True) 
df_note_attr_eff = pd.merge(df_note_attr_eff, temp_note_attr_tot_conc, left_index=True, right_index=True)

df_note_attr_eff.sort_index(inplace=True)

# en pourcentage #
# Sans attrition #
df_note_attr_pct = pd.DataFrame(comments_enquete_client.loc[comments_enquete_client['ATTR_CONC']=='SANS']['Note_recommandation'].value_counts(dropna=False, normalize=True))
df_note_attr_pct.rename(columns = {'Note_recommandation': 'tx_attr_sans'}, inplace=True) 

# Taux d'attrition partielle #
temp_note_attr_part_conc = pd.DataFrame(comments_enquete_client.loc[comments_enquete_client['ATTR_CONC']=='PART']['Note_recommandation'].value_counts(dropna=False, normalize=True))
temp_note_attr_part_conc.rename(columns = {'Note_recommandation': 'tx_attr_partielle'}, inplace=True) 
df_note_attr_pct = pd.merge(df_note_attr_pct, temp_note_attr_part_conc, left_index=True, right_index=True)

# Taux d'attrition totale #
temp_note_attr_tot_conc = pd.DataFrame(comments_enquete_client.loc[comments_enquete_client['ATTR_CONC']=='TOT']['Note_recommandation'].value_counts(dropna=False, normalize=True))
temp_note_attr_tot_conc.rename(columns = {'Note_recommandation': 'tx_attr_totale'}, inplace=True) 
df_note_attr_pct = pd.merge(df_note_attr_pct, temp_note_attr_tot_conc, left_index=True, right_index=True)

df_note_attr_pct.sort_index(inplace=True)
