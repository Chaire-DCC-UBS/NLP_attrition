#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:43:12 2023

@author: ac
"""

# VARIABLES ANALYSÉES #
###############################
APP_VAR_ATTR    = 'ATTR_CONC'                       # attrition #
APP_VAR_VBT     = 'var_lemma_nosw_txt'              # verbatim #

# LISTE DES MODALITÉS D'ATTRITION POUR POUVOIR BOUCLER SUR CETTE LISTE ENSUITE #
################################################################################
modalites_attrition = comments[APP_VAR_ATTR].unique().tolist()

# NUAGE DE MOTS EN FONCTION DE L'ATTRITION #
############################################
for i in modalites_attrition :
    print("Attrition analysée : "+ i)
    df_wc = comments.loc[comments[APP_VAR_ATTR]==i]
    text=df_wc['var_lemma_nosw']
    wordcloud = WordCloud(width=480, height=480, margin=0,prefer_horizontal=1,max_words=50).generate(str(text))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()
    
    

    

# MOTS LES PLUS FRÉQUENTS ET DISCRIMINANTS EN FONCTION DE L'ATTRITION #
#######################################################################
liste_df_attr = []

# CREATION DES INDICATEURS UTILES PAR GROUPE D'ATTRITION #
for i in modalites_attrition :
    df_vocabulary_temp = NLP_FCT.vocab(comments.loc[comments[APP_VAR_ATTR]==i],'var_lemma_nosw',0)
   
    # calcul du nombre de mots totaux par groupe #
    eff_tot = df_vocabulary_temp['effectif'].sum(axis=0)  
    df_vocabulary_temp['freq_w_k'] = df_vocabulary_temp['effectif'] / eff_tot
       
    new_col_name_eff_w_k = 'effectif_word_' + str(i)
    df_vocabulary_temp = df_vocabulary_temp.rename(columns={'effectif': new_col_name_eff_w_k})
    
    new_col_name_eff_tot_k = 'freq_word_' + str(i)
    df_vocabulary_temp = df_vocabulary_temp.rename(columns={'freq_w_k': new_col_name_eff_tot_k})
       
    df_name = f'df_vocabulary_attr_{i}'
    globals()[df_name] = df_vocabulary_temp
    liste_df_attr.append(df_vocabulary_temp)
    
# CREATION DES INDICATEURS UTILES TOUS GROUPES CONFONDUS #
eff_tot_ALLGROUPS = df_vocabulary['effectif'].sum(axis=0)
df_vocabulary['freq_word_ALLGROUPS'] = df_vocabulary['effectif'] / eff_tot_ALLGROUPS
liste_df_attr.append(df_vocabulary)

# FUSION DE L'ENSEMBLE DES INDICATEURS #
df_vocabulary_attr = liste_df_attr[0] 
for df in liste_df_attr[1:] :
    df_vocabulary_attr = pd.merge(df_vocabulary_attr, df, on='word', how='outer')  


# TEST D'ÉGALITÉ DE PROPORTION PAR MOT #
from scipy.stats import chi2_contingency
for i in range(len(modalites_attrition)):
    for j in range(i+1, len(modalites_attrition)):        
        df_vocabulary_attr['chi2_'+ modalites_attrition[i] + '_'+ modalites_attrition[j]], df_vocabulary_attr['p_'+ modalites_attrition[i] + '_' + modalites_attrition[j]], df_vocabulary_attr['dof_'+ modalites_attrition[i] + '_'+ modalites_attrition[j]], df_vocabulary_attr['expected_'+ modalites_attrition[i] + '_'+ modalites_attrition[j]] = zip(*df_vocabulary_attr.apply(lambda row: chi2_contingency([[row['freq_word_'+ modalites_attrition[i]], 1-row['freq_word_'+ modalites_attrition[i]]], [row['freq_word_'+ modalites_attrition[j]], 1-row['freq_word_'+ modalites_attrition[j]]]]), axis=1))
        

# CALCUL DE LA RELEVANCE PAR MOT ET PAR GROUPE #
def relevance(p_w_k, p_w, mylambda):
    relevance_w = mylambda*np.log(p_w_k) + (1-mylambda)*np.log(p_w_k / p_w)
    return(relevance_w)
# avec :       
# p_w_k : fréquence du mot w dans le groupe k / nombre total de mot dans le groupe k
# p_w   : frequence du mot w dans le corpus total / nombre total de mots dans le corpus total
    
for i in modalites_attrition :
    var_freq_word = 'freq_word_' + str(i)
    df_vocabulary_attr['relevance_word_0'] = df_vocabulary_attr.apply(lambda row :                          relevance(row[var_freq_word],row['freq_word_ALLGROUPS'],0), axis=1) 
    df_vocabulary_attr['relevance_word_0_6'] = df_vocabulary_attr.apply(lambda row :                          relevance(row[var_freq_word],row['freq_word_ALLGROUPS'],0.6), axis=1) 
    df_vocabulary_attr['relevance_word_1'] = df_vocabulary_attr.apply(lambda row :                          relevance(row[var_freq_word],row['freq_word_ALLGROUPS'],1), axis=1) 

    new_col_name_relevance_w_k_0 = 'relevance_word_' + str(i) + '_0'
    new_col_name_relevance_w_k_0_6 = 'relevance_word_' + str(i) + '_0-6'
    new_col_name_relevance_w_k_1 = 'relevance_word_' + str(i) + '_1'
    df_vocabulary_attr = df_vocabulary_attr.rename(columns={'relevance_word_0': new_col_name_relevance_w_k_0})   
    df_vocabulary_attr = df_vocabulary_attr.rename(columns={'relevance_word_0_6': new_col_name_relevance_w_k_0_6})    
    df_vocabulary_attr = df_vocabulary_attr.rename(columns={'relevance_word_1': new_col_name_relevance_w_k_1})  
    
     
# TOP POUR DÉTERMINER LA PRÉSENCE D'UN MOT DANS UN VERBATIM #
#############################################################
list_words_to_check = ['sinistre', 'tres']
list_var_words_to_check = []
for word in list_words_to_check:
    nom_var = "presence_" + word
    comments[nom_var] = comments['var_lemma_nosw_txt'].str.contains(word, case=False)
    list_var_words_to_check.append(nom_var)




    
    
