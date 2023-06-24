#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:06:12 2022

@author: ac
"""

# Bibliothèques #
import pandas as pd


# CHARGEMENT DES DONNÉES #
comments_allianz = pd.read_csv('data_tests/data_attrition_allianz.csv', sep=';', encoding='utf-8')

##############################################################################
#                 MISE EN FORME DES BASES D'ÉTUDE                            #
##############################################################################

# Structure de la base d'origine #
print('Nombre de lignes   : ' + repr(comments_allianz.shape[0]) + '\n' + 'Nombre de colonnes : ' + repr(comments_allianz.shape[1]))

# Liste des variables #
variables = comments_allianz.columns.values.tolist()

# Top souscription contrat 12 mois #
comments_allianz.loc[comments_allianz['NBC_12M']==0,'top_NBC_12M'] = '0'
comments_allianz.loc[comments_allianz['NBC_12M']>0,'top_NBC_12M'] = '1'

###############################################################################
#                         QUELQUES STATS DESCRIPTIVES                         #
###############################################################################

# CLIENTS #
###########

# Nombre de clients uniques interrogés #
nb_clients = len(comments_allianz['Id_Client'].unique())

clients = comments_allianz['Id_Client'].value_counts()

test = comments_allianz.loc[comments_allianz['Id_Client'] == 'DWHC_2688310']



# ENQUETES #
############
# Nombre d'enquêtes suivies d'une résiliation dans les 12 mois #
nb_enquete_with_resil_eff = comments_allianz['Attrition'].value_counts(dropna=False)
nb_enquete_with_resil_pct = comments_allianz['Attrition'].value_counts(dropna=False, normalize=True)

# Nombre d'enquêtes suivies d'une souscription dans les 12 mois #
nb_souscr_eff = comments_allianz['top_NBC_12M'].value_counts(dropna=False)
nb_souscr_pct = comments_allianz['top_NBC_12M'].value_counts(dropna=False, normalize=True)






































