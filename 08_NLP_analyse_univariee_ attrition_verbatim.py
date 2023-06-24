#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:59:48 2023

@author: ac
"""


from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_text

###############################################################################
#                               PARAMÉTRAGE                                   # ###############################################################################



# DÉTERMINATION DE LA VARIABLE A EXPLIQUER #
# THELEM - CHOIX 1 - ATTRITION TOT vs SANS #
# (suppression de l'attrition partielle)   #
VarToPredict = 'ATTR_CONC'
list_modalites_y = ['TOT','SANS']
df = comments.loc[comments[VarToPredict].isin(list_modalites_y)]

# THELEM - CHOIX 2  #
comments['ATTR_CONC_TOTPART_SANS'] = comments['ATTR_CONC'].apply(lambda x: 'TOT_OU_PART' if x in ['TOT', 'PART'] else 'SANS')
VarToPredict = 'ATTR_CONC_TOTPART_SANS'
df = comments.copy()

# ALLIANZ - A STATUER #


# TAILLE ÉCHANTILLON VALIDATION #
test_size = 0.3 

#ECRETAGE DES VARIABLES CONTINUES
outlier_mode = 'std'                # 2 choix : iqr (intervalle interquartile) ou std (écart-type)
coeff_ecart = 5                     # Nombre d'écarts interquartile acceptés ou Nbre d'écart-types à la moyenne toléré

# MODALITES RARES
pct_mode_max = 0.95
eff_rare = 200 
LowN = 20                           # Effectif en dessous duquel on considère la modalité comme rare

#VALEURS MANQUANTES
pct_nan_max = 0.01                  # % de valeurs non manquantes pour garder la colonne
#Nan_str = "_MISSING_"              # Valeur de remplacement pour les valeurs manquantes des variables de type "object"
Nan_num = -1                        # Valeur de remplacement pour les valeurs manquantes des variables numériques 
S_Nan_max_for_cont = 0.1            # Proportion max de valeurs manquantes au-delà de laquelle les variables continues sont automatiquement discrétisées avec modalité spécifique pour les données manquantes
Pct_val = 1-pct_nan_max             # S_val: on supprime les colonnes dont une valeur est présente plus de S_val en frequence
#prepro_missing_dummy = True        # Création d'une indicatrice indiquant si la variable contient des données manquantes pour les variables continues

#TYPE DE VARIABLES
nb_level_num = 100                  # Nombre de modalités en dessous duquel on considère une variable numérique comme pouvant être une nominale

#PROBAS, TESTS
Seuil_Proba_chi2 = 0.2              # Seuil_Proba_chi2 : seuil au dessus duquel on supprime la variable si elle est catégorielle
Seuil_Proba_fisher = 0.2            # Seuil au-delà duquel on discrétise la variable si elle est continue

#DISCRETISATION PAR ARBRE DE DECISION
my_seed = 123                       # Graine d'initialisation des tirages aléatoires pour le partitionnement et les algos de Machine learning
state = np.random.RandomState(my_seed)


################################################################################
#                         VARIABLES INUTILES                                   #
################################################################################
df.drop(['date_enquete', 'identifiant_client_anonymisé', 'Raisons_note_recommandation', 'Remarques', 'var_lemma_token','var_lemma_nosw', 'top_resil_partiel', 'top_resil_tot', 'top_attr_part_conc', 'top_attr_tot_conc', 'w2v_vector_mean_unigram', 'w2v_vector_mean_bigram', 'w2v_vector_meantfidf_unigram', 'w2v_vector_meantfidf_bigram'], axis=1, inplace=True)

# Suppression des colonnes qui contiennent des listes #
df = df.drop(df.applymap(type).eq(list).any(0)[lambda x: x].index, axis=1)


################################################################################
#                          VARIABLE CIBLE                                      #
################################################################################
Y = df[VarToPredict].copy()
X_train, X_test= train_test_split(df, test_size = test_size)
idx_train = X_train.index.tolist()


################################################################################
#                     CREATION DES METADONNEES                                 #
################################################################################

variables                   = df.columns.values.tolist()
metadata                    = pd.DataFrame(index=variables)

# STATISTIQUES DESCRIPTIVES
metadata['Variable']        = variables                                                                         # Nom des variables
metadata['Num_Var']         = range(len(variables))                                                             # Attribution d'un numéro à chaque variable
metadata['Type']            = df.dtypes.astype(str)                                                             # Type de variables
metadata['Nb_No_Missing']   = df.shape[0]-df.isnull().sum()                                                     # Nombre de valeurs non manquantes
metadata['Nb_Missing']      = df.isnull().sum()                                                                 # Nombre de valeurs manquantes
metadata['%_Missing']       = df.isnull().sum()/df.shape[0]                                                     # Pourcentage de valeurs manquantes
metadata['Nb_Valeurs']      = df.apply(lambda x:len(x.unique()))                                                # Nombre de modalités par variable         
var_mode_ok                 = metadata.loc[metadata['Nb_Valeurs']<df.shape[0],'Variable'].tolist()              
# Liste des variables avec un mode
modestmp                    = df[var_mode_ok].mode()                                                            # Calcul du mode pour chaque variable avec mode
list_modes                  = modestmp.iloc[0]                                                                  # Récupération du mode pour chaque variable avec mode
metadata.loc[modestmp.columns,'Mode']                   = list_modes
metadata.loc[modestmp.columns,'Nb_modes_distincts']     = modestmp.apply(pd.Series.nunique)

for var in modestmp.columns :
    mode_var = metadata.loc[var,'Mode']
    metadata.loc[var,'Freq_Mode']=(df[var]==mode_var).astype(int).sum()                                         # Fréquence de la valeur modale
metadata['Pct_Mode']=metadata['Freq_Mode']/metadata['Nb_No_Missing']                                            # Proportion de la valeur modale

# ANALYSES DES VARIABLES CONTINUES
var_num = df.select_dtypes(exclude=['object','bool','category','datetime']).columns.values.tolist()             # Liste des variables continues

if len(var_num)>0:
    meta_num = df[var_num].describe(percentiles=[.1,.25,.5,.75,.9]).transpose().iloc[:,1:]                      # Stats descriptives classiques sur variables continues
    meta_num.columns = ['Mean','Std','Min','D1','Q1','Median','Q3','D9','Max']                                  # Renommage des variables de stats
    meta_num['InterQuartile']=meta_num['Q3']-meta_num['Q1']                                                     # Intervalle InterQuartile
    if outlier_mode == 'iqr' :                                                                                  # Calcul des bornes pour ecretage
        meta_num['Outlier_Left']    = meta_num['Median'] - coeff_ecart*meta_num['InterQuartile']
        meta_num['Outlier_Right']   = meta_num['Median'] + coeff_ecart*meta_num['InterQuartile']
    else :
        meta_num['Outlier_Left']    = meta_num['Median'] - coeff_ecart*meta_num['Std']
        meta_num['Outlier_Right']   = meta_num['Median'] + coeff_ecart*meta_num['Std']
    meta_num['Nb_Outlier_Left']  = [(df[x].dropna()<meta_num.loc[x,'Outlier_Left']).sum() for x in var_num]
    meta_num['Nb_Outlier_Right'] = [(df[x].dropna()>meta_num.loc[x,'Outlier_Right']).sum() for x in var_num]   
    meta_num['Nb_Outlier']       = meta_num['Nb_Outlier_Left'] + meta_num['Nb_Outlier_Right']
    metadata = pd.concat([metadata,meta_num],axis=1)
else :
    metadata['Mean']=np.nan
    metadata['Std']=np.nan   
    metadata['Min']=np.nan   
    metadata['D1']=np.nan   
    metadata['Q1']=np.nan   
    metadata['Median']=np.nan  
    metadata['Q3']=np.nan     
    metadata['D9']=np.nan
    metadata['Max']=np.nan
    metadata['InterQuartile']=np.nan
    metadata['Outlier_Left']=np.nan
    metadata['Outlier_Right']=np.nan
    metadata['Nb_Outlier_Left']=np.nan
    metadata['Nb_Outlier_Right']=np.nan
    metadata['Nb_Outlier']=np.nan
     



# VARIABLES PARTICULIERES
metadata['Commentaires']=np.nan 
metadata.loc[metadata['Pct_Mode']>pct_mode_max,'Commentaires']                          = 'Constante ou quasi-constante'
metadata.loc[metadata['Nb_No_Missing'] < int(df.shape[0]*pct_nan_max),'Commentaires']   = 'Trop de valeurs manquantes'
metadata['Var_a_Supprimer']= (metadata['Pct_Mode']>pct_mode_max) | (metadata['Nb_No_Missing'] < int(df.shape[0]*pct_nan_max))




# ROLE DES VARIABLES
metadata.loc[metadata['Type'].isin(['object','bool','category']),'Role']                                    = 'Categorielle'
metadata.loc[metadata['Type'].isin(['float32','float64','int8', 'int16','int32','int64','uint8']),'Role']   = 'Continue'
metadata.loc[metadata['Var_a_Supprimer'],'Role']                                                            = 'ASupprimer'
metadata.loc[metadata['Nb_Valeurs'] < nb_level_num,'Role']                                                  = 'Categorielle'
metadata.loc[metadata['Nb_Valeurs']==1,'Role']                                                              = 'Unaire'
metadata.loc[metadata['Nb_Valeurs']==2,'Role']                                                              = 'Binaire'



# ANALYSES DES VARIABLES CATEGORIELLES
# Modalités rares
var_cat = metadata.loc[metadata['Role'] == 'Categorielle']['Variable'].tolist()
if len(var_cat) > 0 :
    listmod = pd.DataFrame()
    listmod['List_mod_rares'] = df[var_cat].apply(lambda x: (x.value_counts()[x.value_counts()<eff_rare].index.tolist()))  
    listmod['Nb_mod_rares'] = listmod['List_mod_rares'].apply(lambda x : len(x))
    metadata = pd.concat([metadata, listmod], axis=1)


# Liaisons entre variables explicatives et variable à expliquer
meta_cat = metadata.loc[(metadata['Role'] == 'Binaire') | (metadata['Role'] == 'Categorielle')]
if not (Y is None) :
    list_var_cat = meta_cat['Variable'].tolist()
    if len (list_var_cat) > 0 :
        e=df.loc[idx_train, list_var_cat].copy()
        data=[]
        
        for var in var_cat :
            # Remplacement des données manquantes
            if meta_cat.loc[var,'Type'] == ['object'] :
                e[var].fillna(Nan_str,inplace=True)
            elif meta_cat.loc[var,'Type'] == ['category'] :
                if metadata.loc[var,'Nb_Missing']>0 : e[var] = e[var].cat.add_categories(Nan_str).fillna(Nan_str) # cas particulier des variables "categories"
            else :
                e[var].fillna(Nan_num,inplace=True)
                
            # Calcul du pouvoir prédictif des variables catégorielles
            Y_train = X_train[VarToPredict]
            tab=pd.crosstab(e[var],Y_train)
            chi2, pchi2 = chi2_contingency(tab,correction=False)[0:2]   # on ne retient que Chi2 et proba chi2
            nr,nc = tab.shape                                           # récupération des dimensions pour calcul des ddl
            t = np.sqrt(chi2/df.shape[0]/np.sqrt((nr-1)*(nc-1)))        # T de Tschuprow
            v = np.sqrt((chi2)/(df.shape[0]*(min(nr,nc)-1)))
            data.append([var,chi2,pchi2,max(nr,nc)-1,t,v])

        metadataChi2 = pd.DataFrame(data=data,columns=['Variable','Chi2','P_Chi2','Nb_dll','Tschuprow','V_Cramer'])
        metadataChi2 = metadataChi2.set_index('Variable')
        metadata = pd.concat([metadata,metadataChi2],axis=1)  
    else :
        metadata['List_mod_rares']  = np.nan
        metadata['Nb_mod_rares']    = np.nan
        metadata['Nb_dll']          = np.nan
        metadata['Chi2']            = np.nan
        metadata['P_Chi2']          = np.nan
        metadata['Tschuprow']       = np.nan       
        metadata['V_Cramer']        = np.nan        
        
        
    # LIAISONS ENTRE VARIABLE A EXPLIQUER ET VARIABLES CONTINUES
    meta_cont = metadata.loc[(metadata['Variable'].isin(var_num))&(pd.notnull(metadata['Role']))]
    var_cont = meta_cont['Variable'].tolist()
    if len(var_cont) > 0 :
        e = df.loc[idx_train,var_cont].copy()
        
        # Remplacement des données manquantes
        e = e.apply(lambda x: x.fillna(x.mean()),axis=0)
        
        # Suppression des constantes
        nb_v_dist = e.apply(lambda x : len(x.unique())) # Calcul du nombre de valeurs distinctes par variable
        liste_cst = list(nb_v_dist[nb_v_dist==1].index) # Liste des index correspondant à des constantes
        e.drop(liste_cst,axis=1,inplace=False)          # Suppression des constantes
        
        # Calcul du pouvoir prédictif des variables continues
        anova = SelectKBest(f_classif,k='all')
        anova.fit_transform(e,Y.loc[idx_train])
        anova_features = sorted(enumerate(anova.scores_),key=lambda x:x[0])
        proba_features = sorted(enumerate(anova.pvalues_),key=lambda x:x[0])
        
        metadataanova = pd.DataFrame(index=e.columns.values.tolist())
        
        metadataanova['F_Fisher'] = [x[1] for x in anova_features]
        metadataanova['P_F_Fisher'] = [x[1] for x in proba_features]
        
        for v in liste_cst :                            # Réintégration des variables constantes
            metadataanova.loc[v] = [0, 1]
            
        metadata = pd.concat([metadata, metadataanova], axis=1)        
        
        
        # LIAISONS ENTRE VARIABLE A EXPLIQUER ET VARIABLES CONTINUES ECRETEES
        meta_cont_rob = meta_cont.loc[meta_cont['Nb_Outlier'] > 0 & (meta_cont['Q1']!=meta_cont['Q3']) & (meta_cont['Var_a_Supprimer'] == 0)]
        var_cont_rob = meta_cont_rob['Variable'].tolist() 

        if len(var_cont_rob)>0 :
            f=df.loc[idx_train,var_cont_rob].copy()
            f = f.apply(lambda x: x.fillna(x.mean()),axis=0)
            # Ecretage
            LL = meta_cont_rob['Outlier_Left']
            UL = meta_cont_rob['Outlier_Right']
            f = pd.DataFrame(np.where(f > UL, UL, np.where(f < LL, LL, f)), columns=var_cont_rob)
            # Calcul pouvoir explicatif
            anovaR = SelectKBest(f_classif,k='all')
            anovaR.fit_transform(f, Y.loc[idx_train])
            anovaR_features = sorted(enumerate(anovaR.scores_),key=lambda x:x[0])
            probaR_features = sorted(enumerate(anovaR.pvalues_),key=lambda x:x[0])
        
            metadataanovaR = pd.DataFrame(index = f.columns.values.tolist())
            metadataanovaR['F_Fisher_R'] = [x[1] for x in anovaR_features]
            metadataanovaR['P_F_Fisher_R'] = [x[1] for x in probaR_features]
            metadata = pd.concat([metadata, metadataanovaR], axis=1)
        else :
            metadata['F_Fisher_R'] = np.nan
            metadata['P_F_Fisher_R'] = np.nan
        
    else :  # variables non continues
        metadata['F_Fisher'] = np.nan
        metadata['P_F_Fisher'] = np.nan
        metadata['P_F_Fisher_R'] = np.nan        
        metadata['F_Fisher_R'] = np.nan


# ROLE FINAL DES VARIABLES
metadata.loc[metadata['Type'].isin(['float32','float64','int16','int32','int8', 'int64', 'uint8','uint16']),'Role_Final']    = 'Continue'
metadata.loc[metadata['Type'].isin(['float32','float64','int16','int32','int8','int64','uint8','uint16']) & (metadata['Nb_Valeurs'] < nb_level_num),'Role_Final'] = 'Cat_num'
metadata.loc[metadata['Type'].isin(['float32','float64','int16','int32','int64','int8','uint8','uint16']) & (metadata['P_F_Fisher'] > Seuil_Proba_fisher) ,'Role_Final'] = 'A_discretiser'
metadata.loc[metadata['Type'].isin(['float32','float64','int16','int32','int64','int8','uint8','uint16']) & (metadata['%_Missing'] > S_Nan_max_for_cont) ,'Role_Final'] = 'A_discretiser'
metadata.loc[metadata['Type'].isin(['float32','float64','int16','int32','int8','int64','uint8','uint16']) & (metadata['Max'] > metadata['Outlier_Right']) & (meta_cont['Q1']!=meta_cont['Q3']) ,'Role_Final'] = 'A_ecreter'     # on rajoute la condition Q1 différent de Q3 pour éviter d'écréter des variables avec peu de valeurs différentes, qui peuvent donc devenir constante si on écrète
metadata.loc[metadata['Type'].isin(['float32','float64','int16','int32','int8','int64','uint8','uint16']) & (metadata['Min'] < metadata['Outlier_Left']) & (meta_cont['Q1']!=meta_cont['Q3']) ,'Role_Final'] = 'A_ecreter'      # même remarque que précédemment pour Q1 différent de Q3.

metadata.loc[metadata['Type'].isin(['object','bool','category']),'Role_Final'] = 'Cat_str'
metadata.loc[metadata['Type'].isin(['object','bool','category']) & (metadata['P_Chi2'] > Seuil_Proba_chi2) ,'Role_Final'] = 'A_supprimer'
# VOIR COMMENT OPTIMISER LE CODE !!!!!!!!!!!!!!!!!!!
metadata.loc[metadata['Type'].isin(['object','bool','category']) & (metadata['P_Chi2'] > Seuil_Proba_chi2) ,'Commentaires'] = 'Lien trop faible avec la variable à expliquer'


metadata.loc[metadata['Pct_Mode']>Pct_val,'Role_Final'] = 'A_Supprimer'      # Variables constantes ou quasi constantes
metadata.loc[metadata['Pct_Mode']>Pct_val,'Commentaires'] = 'Constante ou quasi-constante'

metadata.loc[metadata['%_Missing']>pct_nan_max,'Role_Final'] = 'A_Supprimer' # Variables avec trop fort taux de valeurs manquantes
metadata.loc[metadata['%_Missing']>pct_nan_max,'Commentaires'] = 'Trop de valeurs manquantes'

metadata.loc[(metadata['Role_Final']=='Cat_num') & (metadata['P_Chi2'] > Seuil_Proba_chi2) & (metadata['P_F_Fisher'] > Seuil_Proba_fisher) & (metadata['P_F_Fisher_R'] > Seuil_Proba_fisher),'Role_Final'] = 'A_supprimer' # Variables catégorielles numériques avec faible lien avec la variable à expliquer
metadata.loc[(metadata['Role_Final']=='Cat_num') & (metadata['P_Chi2'] > Seuil_Proba_chi2) & (metadata['P_F_Fisher'] > Seuil_Proba_fisher) & (metadata['P_F_Fisher_R'] > Seuil_Proba_fisher),'Commentaires'] = 'Lien trop faible avec la variable à expliquer' # Variables catégorielles numériques avec faible lien avec la variable à expliquer

metadata.loc[metadata['Nb_Valeurs']==df.shape[0],'Role_Final'] = 'A_Supprimer' # Variables avec autant de valeurs distinctes que de lignes - identifiant ?
metadata.loc[metadata['Nb_Valeurs']==df.shape[0],'Commentaires'] = 'Variable supposée identifiant'

metadata.loc[metadata['V_Cramer']==1,'Role_Final'] = 'A_Supprimer' # Variables avec autant de valeurs distinctes que de lignes - identifiant ?
metadata.loc[metadata['V_Cramer']==1,'Commentaires'] = 'Variable prédisant parfaitement la variable à expliquer'



################################################################################
#                      GESTION DES VALEURS MANQUANTES                          # 
################################################################################

# CAS 1 - variables continues avec valeurs manquantes :
metadata.loc[metadata['Type'].isin(['float32','float64','int16','int32','int8','uint8','uint16']) & (metadata['%_Missing'] > 0) & (metadata['Role_Final'].isin(['Continue','A_discretiser'])) ,'Traitement_VM'] = 'Continue_Mean'
# CAS 2 - variables continues avec valeurs manquantes, à écréter :
metadata.loc[metadata['Type'].isin(['float32','float64','int16','int32','int8','uint8','uint16']) & (metadata['%_Missing'] > 0) & (metadata['Role_Final'] == 'A_ecreter') ,'Traitement_VM'] = 'Continue_Median' 

meta_avec_vm_a_gerer = metadata.loc[metadata['Traitement_VM'].isin(['Continue_Mean','Continue_Median'])]
list_var_avec_vm_a_gerer = meta_avec_vm_a_gerer['Variable'].tolist()

for var in list_var_avec_vm_a_gerer :
    new_var = metadata.loc[var,'Mean']
    idx_vm = df[var].isnull()
    df.loc[idx_vm, var] = new_var


################################################################################
#            SUPPRESSION DES VARIABLES SANS POUVOIR EXPLICATIF                 #
################################################################################
metadata_del    = metadata.loc[metadata['Role_Final']=='A_Supprimer']
metadata_varok  = metadata.loc[metadata['Role_Final']!='A_Supprimer']


################################################################################
#          RÉPARTITION MODALITÉS VARIABLES EXPLICATIVES (TRAIN + TEST)         #
################################################################################
list_var_cat = metadata.loc[(metadata['Role_Final'].isin(['Cat_num', 'Cat_str', 'Dummy'])), 'Variable'].tolist()
list_var_cont = metadata.loc[metadata['Role_Final'].isin(['Continue', 'A_discretiser', 'A_ecreter']), 'Variable'].tolist()

# STATS VARIABLES CATÉGORIELLES - TRAIN+TEST#
dict_modalites_varcat = {}
for var in list_var_cat :
    dict_modalites_varcat[var] = pd.crosstab(index=df[var], columns='count')
    

################################################################################
#                           STATS VARIABLE CIBLE                               #
################################################################################
dict_modalites_var_by_ytrain = {}
for var in list_var_cat :
    dict_modalites_var_by_ytrain[var] = pd.crosstab(index=df[var], columns=Y_train, margins=True)

#for var in list_var_cont :    
for var in ['Age'] :
    disc_var_name = var + '_disc'
    df[disc_var_name] = pd.qcut(df[var], q=20)
    dict_modalites_var_by_ytrain[var] = pd.crosstab(index=df[disc_var_name], columns=Y_train, margins=True)


################################################################################
#                       ÉCRÉTAGE DES VARIABLES CONTINUES                       #  
################################################################################
meta_a_ecreter = metadata_varok.loc[metadata_varok['Role_Final']=='A_ecreter'] 

list_var_a_ecreter = meta_a_ecreter['Variable'].tolist()
if len (list_var_a_ecreter) > 0 :       
    for var in list_var_a_ecreter :
        val_left = metadata.loc[var,'Outlier_Left']
        val_right = metadata.loc[var,'Outlier_Right']
        idx_left = df[var] < val_left
        idx_right = df[var] > val_right
        df.loc[idx_left, var] = val_left
        df.loc[idx_right, var] = val_right


################################################################################
#                       VARIABLES CORRÉLÉES                                    #  
################################################################################
        
## CORRÉLATION DES VARIABLES CONTINUES #
#df_corr_cont = pd.DataFrame(columns=['var_cont_1', 'var_cont_2', 'Coeff_Pearson', 'p-value'])
#for var1 in list_var_cont :
#    for var2 in list_var_cont :
#        coeff_pearson, p_val = pearsonr(df[var1],df[var2])
#        new_row = {'var_cont_1': var1, 'var_cont_2': var2, 'Coeff_Pearson': coeff_pearson, 'p-value': p_val}
#        df_corr_cont = pd.concat([df_corr_cont, pd.DataFrame(new_row, index=[0])], ignore_index=True)
#test = df[list_var_cont]
#
## CORRÉLATION DES VARIABLES CATÉGORIELLES #
#df_corr_cat = pd.DataFrame(columns=['var_cat_1', 'var_cat_2', 'Chi2', 'p-value'])
#for var1 in list_var_cat :
#    for var2 in list_var_cat :
#        contingency_table = pd.crosstab(df[var1], df[var2])
#        #chi2, pchi2 = chi2_contingency(contingency_table,correction=False)[0:2]
#        #new_row = {'var_cat_1': var1, 'var_cat_2': var2, 'Chi2': chi2, 'p-value': pchi2}
#        #df_corr_cat = pd.concat([df_corr_cat, pd.DataFrame(new_row, index=[0])], ignore_index=True)



################################################################################
#            DISCRETISATION SUPERVISÉE DES VARIABLES CONTINUES                 #  
################################################################################
 
# REMPLACEMENT DES VALEURS MANQUANTES # 
#list_var_a_discretiser = metadata.loc[metadata['Role_Final']== 'A_discretiser', 'Variable'].tolist()
list_var_a_discretiser = ['Age']
for var in list_var_a_discretiser :
    # Remplacement des valeurs manquantes des variables continues par la moyenne #  
    print(var)
    mean_var = df[var].mean()
    print ("Valeurs manquantes de " + var + " remplacées par " + str(round(mean_var)))
    df[var+'_remplacevm'] = df[var].fillna(mean_var)  
    X_var = df[var+'_remplacevm'].values.reshape(-1,1)    
    
    

    
################################################################################
#                           ONE-HOT-ENCODING                                   #  
################################################################################
list_var_a_onehotencoder = ['type_enquete']
encoder = OneHotEncoder()
list_var_apres_encodage = []  # liste pour stocker les noms des nouvelles variables créées par l'encodage #
for var in list_var_a_onehotencoder :
    new_cols = pd.get_dummies(df[var], prefix=var)
    df = pd.concat([df, new_cols], axis=1)
    list_var_apres_encodage += list(new_cols.columns)
    
    
    

















