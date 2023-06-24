#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:19:01 2022

@author: ac
"""
from nltk import FreqDist

APP_VAR_VBT = 'Raisons_note_recommandation'
#APP_VAR = 'Remarques'

#APP_VAR = 'Recommandation_amelioration'
#APP_VAR = 'Recommandation_satisfaction'
#APP_VAR = 'Recommandation_commentaire'
#APP_VAR = 'Echange_amelioration'
#APP_VAR = 'Echange_satisfaction'
#APP_VAR = 'Echange_commentaire'

comments = comments.dropna(subset=[APP_VAR_VBT])

# NOMBRE DE MOTS PAR VERBATIM #
###############################
comments['nombre_mots'] = comments['var_lemma_token'].apply(lambda mylist : len(mylist))

# PONCTUATION PAR VERBATIM #
############################
# Nombre de points d'exclamation #
comments['nbr_exclam']      = comments[APP_VAR_VBT].apply(lambda text : FreqDist(str(text))['!'])

# Nombre de points d'interrogation #
comments['nbr_interrog']    = comments[APP_VAR_VBT].apply(lambda text : FreqDist(str(text))['?'])
# Nombre de points d'exclamation + d'interrogation #
comments['nbr_exclam_interrog'] = comments['nbr_exclam'] + comments['nbr_interrog']

# Top présence points d'exclamation #
comments.loc[comments['nbr_exclam']==0,'top_exclam'] = '0'
comments.loc[comments['nbr_exclam']>0,'top_exclam'] = '1'

# Top présence points d'interrogation #
comments.loc[comments['nbr_interrog']==0,'top_interrog'] = '0'
comments.loc[comments['nbr_interrog']>0,'top_interrog'] = '1'


# NOMBRE D'ADJECTIFS PAR VERBATIM #
###################################
nlp = spacy.load('fr_core_news_sm')
def count_adjectives(text):
    # Analyse le texte avec le modèle de langue de spaCy
    doc = nlp(text)
    # Compte le nombre d'adjectifs dans le texte
    count = sum(1 for token in doc if token.pos_ == 'ADJ')
    return count

comments['nb_adjectifs'] = comments[APP_VAR_VBT].apply(count_adjectives)



# STATISTIQUES DESCRIPTIVES : LONGUEUR DU VERBATIM ET DE LA PONCTUATION #
# Nombre moyen de mots #
nb_mots = comments['nombre_mots'].mean()

# Nombre moyen de mots en fonction de l'attrition #
nb_mots_attrition = comments.groupby(['ATTR_CONC'], as_index=False)['nombre_mots'].mean()
# Nombre moyen de points d'exclamation en fonction de l'attrition # 
nb_exclam_attrition = comments.groupby(['ATTR_CONC'], as_index=False)['nbr_exclam'].mean()
# Nombre moyen de points d'interrogation en Attrition de l'attrition # 
nb_interrog_attrition = comments.groupby(['ATTR_CONC'], as_index=False)['nbr_interrog'].mean()
# Nombre moyen de points d'exclamation et d'interrogation en fonction de l'attrition # 
nb_exclam_interrog_attrition = comments.groupby(['ATTR_CONC'], as_index=False)['nbr_exclam_interrog'].mean()

# Taux de verbatim avec au moins un point un point d'exclamation / d'interrogation #
top_exclam_eff = comments['top_exclam'].value_counts(dropna=False)
top_exclam_pct = comments['top_exclam'].value_counts(dropna=False, normalize=True)

top_interrog_eff = comments['top_interrog'].value_counts(dropna=False)
top_interrog_pct = comments['top_interrog'].value_counts(dropna=False, normalize=True)

# Taux d'attrition en fonction de la présence d'au moins un point d'exclamation / d'interrogation #
attr_exclam_eff = comments.groupby(['top_exclam'], as_index=False)['ATTR_CONC'].value_counts(dropna=False)
attr_exclam_pct = comments.groupby(['top_exclam'], as_index=False)['ATTR_CONC'].value_counts(dropna=False, normalize=True)
attr_interrog_eff = comments.groupby(['top_interrog'], as_index=False)['ATTR_CONC'].value_counts(dropna=False)
attr_interrog_pct = comments.groupby(['top_interrog'], as_index=False)['ATTR_CONC'].value_counts(dropna=False, normalize=True)

































