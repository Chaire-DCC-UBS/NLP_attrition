#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:35:16 2022

@author: Anne-Cécile Gay
"""

# Bibliothèques #
import pandas as pd


# CHARGEMENT DES DONNÉES #
comments_thelem = pd.read_excel('data_tests/data_attrition_thelem.xlsx')


##############################################################################
#                 MISE EN FORME DES BASES D'ÉTUDE                            #
##############################################################################

# Structure de la base d'origine #
print('Nombre de lignes   : ' + repr(comments_thelem.shape[0]) + '\n' + 'Nombre de colonnes : ' + repr(comments_thelem.shape[1]))

# Liste des variables #
variables = comments_thelem.columns.values.tolist()

# CLÉ UNIQUE (ENQUÊTE, CLIENT) : infos enquête et client #
comments_enquete_client = comments_thelem.loc[:,['date_enquete', 'type_enquete', 'identifiant_client_anonymisé', 'Note_recommandation', 'Raisons_note_recommandation', 'Effort', 'Experience', 'Satisfaction_globale', 'Remarques', 'Ancienneté (en année)', 'Age', 'type_client']]
comments_enquete_client.drop_duplicates(inplace = True)

# CLÉ UNIQUE (CLIENT, RESILIATION) : infos résiliations #
comments_client_resiliation = comments_thelem.loc[:,['identifiant_client_anonymisé', 'ID_CTR_RESIL_ANONYMISE', 'DAT_EFF_RSL', 'COD_MTF_RSL', 'LBL_MTF_RSL', 'COD_GRP_MTF_RSL', 'MOTIF', 'top_resil_tot', 'top_resil_partiel']]
comments_client_resiliation.dropna(subset=['ID_CTR_RESIL_ANONYMISE'], inplace = True)
comments_client_resiliation.drop_duplicates(inplace=True)

# CLÉ UNIQUE (ENQUÊTE, CLIENT, RESILIATION) : infos résiliations #
comments_enquete_client_resiliation = comments_thelem.loc[:,['date_enquete', 'type_enquete', 'identifiant_client_anonymisé', 'ID_CTR_RESIL_ANONYMISE', 'DAT_EFF_RSL', 'COD_MTF_RSL', 'LBL_MTF_RSL', 'COD_GRP_MTF_RSL', 'MOTIF', 'top_resil_tot', 'top_resil_partiel']]
comments_enquete_client_resiliation.dropna(subset=['ID_CTR_RESIL_ANONYMISE'], inplace = True)
comments_enquete_client_resiliation.drop_duplicates(inplace=True)

# CREATION D'UN TOP "attrition dans les 12 mois" PAR CLE (ENQUETE, CLIENT) #
df_resil_part = comments_thelem.groupby(['date_enquete', 'type_enquete', 'identifiant_client_anonymisé'], as_index=False)['top_resil_partiel'].max()
df_resil_tot = comments_thelem.groupby(['date_enquete', 'type_enquete', 'identifiant_client_anonymisé'], as_index=False)['top_resil_tot'].max()
df_resil = pd.merge(df_resil_part, df_resil_tot,on=['date_enquete', 'type_enquete', 'identifiant_client_anonymisé'])

comments_enquete_client = pd.merge(comments_enquete_client, df_resil, on=['date_enquete', 'type_enquete', 'identifiant_client_anonymisé'])


# CREATION DE TOPS "attrition partielle et totale dans les 12 mois" PAR CLE (ENQUETE, CLIENT) AVEC MOTIF CONCURRENCE #
comments_thelem['top_attr_part_conc'] = comments_thelem.loc[comments_thelem['COD_GRP_MTF_RSL'] == 'CONC', 'top_resil_partiel']
comments_thelem['top_attr_tot_conc']  = comments_thelem.loc[comments_thelem['COD_GRP_MTF_RSL'] == 'CONC', 'top_resil_tot']
df_resil_part_conc = comments_thelem.groupby(['date_enquete', 'type_enquete', 'identifiant_client_anonymisé'], as_index=False)['top_attr_part_conc'].max()
df_resil_tot_conc = comments_thelem.groupby(['date_enquete', 'type_enquete', 'identifiant_client_anonymisé'], as_index=False)['top_attr_tot_conc'].max()
df_resil_conc = pd.merge(df_resil_part_conc, df_resil_tot_conc, on=['date_enquete', 'type_enquete', 'identifiant_client_anonymisé'])
comments_enquete_client = pd.merge(comments_enquete_client, df_resil_conc, on=['date_enquete', 'type_enquete', 'identifiant_client_anonymisé'])

comments_enquete_client.loc[(comments_enquete_client['top_resil_partiel']!= 1) & (comments_enquete_client['top_resil_tot']!= 1),'ATTR_ALL'] = 'SANS'
comments_enquete_client.loc[comments_enquete_client['top_resil_partiel']== 1,'ATTR_ALL'] = 'PART'
comments_enquete_client.loc[comments_enquete_client['top_resil_tot']== 1,'ATTR_ALL'] = 'TOT'
                           
comments_enquete_client.loc[(comments_enquete_client['top_attr_part_conc']!= 1) & (comments_enquete_client['top_attr_tot_conc']!= 1),'ATTR_CONC'] = 'SANS'
comments_enquete_client.loc[comments_enquete_client['top_attr_part_conc']== 1,'ATTR_CONC'] = 'PART'
comments_enquete_client.loc[comments_enquete_client['top_attr_tot_conc']== 1,'ATTR_CONC'] = 'TOT'






###############################################################################
#                         QUELQUES STATS DESCRIPTIVES                         #
###############################################################################

# CLIENTS #
###########

# Nombre de clients uniques interrogés #
nb_clients = len(comments_thelem['identifiant_client_anonymisé'].unique())

clients = comments_thelem['identifiant_client_anonymisé'].value_counts()

test = comments_thelem.loc[comments_thelem['identifiant_client_anonymisé'] == 'ad9ffcbd1f8f83ab5aad7ea94bb43bb4']
test = comments_thelem.loc[comments_thelem['identifiant_client_anonymisé'] == '062d925d6b76bfffafa4c7337a31f19d']
test = comments_thelem.loc[comments_thelem['identifiant_client_anonymisé'] == '31c8e1cf8ede6f36c688de0440a59272']



# ENQUETES #
############

# Nombre d'enquêtes analysées #
nb_enquetes = len(comments_enquete_client)

# Nombre d'enquêtes suivies d'une résiliation partielle ou totale dans les 12 mois #
nb_enquete_with_resil_part = len(comments_enquete_client.loc[comments_enquete_client['ATTR_ALL'] == 'PART'])
nb_enquete_with_resil_tot  = len(comments_enquete_client.loc[comments_enquete_client['ATTR_ALL'] == 'TOT'])
nb_enquete_with_resil_part_conc = len(comments_enquete_client.loc[comments_enquete_client['ATTR_CONC'] == 'PART'])
nb_enquete_with_resil_tot_conc  = len(comments_enquete_client.loc[comments_enquete_client['ATTR_CONC'] == 'TOT'])
nb_enquete_without_resil = len(comments_enquete_client.loc[comments_enquete_client['ATTR_ALL'] == 'SANS'])



# RÉSILIATIONS #
################

# Nombre de résiliations enregistrées tous motifs confondus #
nb_resil = len(comments_client_resiliation)

# Nombre de résiliations partielles ou totales analysées #
nb_resil_part = len(comments_client_resiliation.loc[comments_client_resiliation['top_resil_partiel'] == 1])
nb_resil_tot  = len(comments_client_resiliation.loc[comments_client_resiliation['top_resil_tot'] == 1])

# Nombre de (enquete, résiliations) uniques tous motifs confondus #
nb_resil_enquete = len(comments_enquete_client_resiliation)

# Nombre de résiliations motif CONCURRENCE #
motifs_resil_eff = comments_client_resiliation['COD_GRP_MTF_RSL'].value_counts()
motifs_resil_pct = comments_client_resiliation['COD_GRP_MTF_RSL'].value_counts(normalize=True)
nb_resil_conc = len(comments_client_resiliation.loc[comments_client_resiliation['COD_GRP_MTF_RSL'] == 'CONC'])

# Nombre de (enquete, résiliations) motif CONCURRENCE #
nb_resil_enquete_conc = len(comments_enquete_client_resiliation.loc[comments_enquete_client_resiliation['COD_GRP_MTF_RSL'] == 'CONC'])

# Nombre de (enquete, résiliations) motif CONCURRENCE entraînant ATTRITION TOTALE #
nb_resil_enquete_conc_attrtot = len(comments_enquete_client_resiliation.loc[(comments_enquete_client_resiliation['COD_GRP_MTF_RSL'] == 'CONC') & (comments_enquete_client_resiliation['top_resil_tot'] == 1)])

# Nombre de (enquete, résiliations) motif CONCURRENCE entraînant ATTRITION PARTIELLE #
nb_resil_enquete_conc_attrpart = len(comments_enquete_client_resiliation.loc[(comments_enquete_client_resiliation['COD_GRP_MTF_RSL'] == 'CONC') & (comments_enquete_client_resiliation['top_resil_partiel'] == 1)])


# SOUSCRIPTIONS #
#################

# Nombre de souscriptions uniques analysées #
nb_souscr = len(comments_thelem['ANONYMISE_ID_CTR_SS'].unique())


# SYNTHESE #
############
indicateurs = ['nb_clients', 'nb_enquetes', 'nb_enquete_without_resil', 'nb_enquete_with_resil_part', 'nb_enquete_with_resil_tot', 'nb_enquete_with_resil_part_conc', 'nb_enquete_with_resil_tot_conc', 'nb_resil', 'nb_resil_part', 'nb_resil_tot', 'nb_resil_conc', 'nb_resil_enquete', 'nb_resil_enquete_conc', 'nb_resil_enquete_conc_attrtot', 'nb_resil_enquete_conc_attrpart', 'nb_souscr']
stats_descs = pd.DataFrame()
stats_descs['stats']   = indicateurs
stats_descs['valeurs'] = [nb_clients, nb_enquetes, nb_enquete_without_resil, nb_enquete_with_resil_part, nb_enquete_with_resil_tot, nb_enquete_with_resil_part_conc, nb_enquete_with_resil_tot_conc, nb_resil, nb_resil_part, nb_resil_tot, nb_resil_conc, nb_resil_enquete, nb_resil_enquete_conc, nb_resil_enquete_conc_attrtot, nb_resil_enquete_conc_attrpart, nb_souscr] 









