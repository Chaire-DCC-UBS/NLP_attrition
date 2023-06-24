#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:57:22 2023

@author: ac
"""
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


APP_VAR_VBT  = 'Raisons_note_recommandation'     # verbatim #



# DÉTERMINATION DE LA VARIABLE A EXPLIQUER #
# THELEM - CHOIX 1 - ATTRITION TOT vs SANS #
# (suppression de l'attrition partielle)   #
list_modalites_y = ['TOT','SANS']
top_attr = 'TOT'
top_no_attr = 'SANS'
VarToPredict = 'ATTR_CONC'

# THELEM - CHOIX 2 - ATTRITION TOT_PART vs SANS #
VarToPredict = 'ATTR_CONC_TOTPART_SANS'
list_modalites_y = ['TOT_OU_PART','SANS']
top_attr = 'TOT_OU_PART'
top_no_attr = 'SANS'




thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

df = df.loc[df[VarToPredict].isin(list_modalites_y)]
df[VarToPredict+'_num'] = df[VarToPredict].map({top_no_attr: 0, top_attr: 1}) # Transformation de la variable à expliquer en indicatrice #

effectif_attrition = df[VarToPredict+'_num'].value_counts()


# TRANSFORMATION DES VARIABLES CATÉGORIELLES EN DUMMIES #
df = pd.get_dummies(df, columns=['label_cluster'])
list_var_dummy = df.filter(like='label_cluster_').columns.tolist()

# LISTE VARIABLES EXPLICATIVES #
var_expl_to_test = [
'nombre_mots',
'nbr_exclam',
'nbr_interrog',
#'nbr_exclam_interrog',
#'top_exclam',
#'top_interrog',
'nb_adjectifs',
'textblob_polarite',
'textblob_subjectivite',
'nombre_mots_positifs_FEEL',
'nombre_mots_negatifs_FEEL',
'nombre_mots_joie_FEEL',
'nombre_mots_peur_FEEL',
'nombre_mots_tristesse_FEEL',
'nombre_mots_colere_FEEL',
'nombre_mots_surprise_FEEL',
'nombre_mots_degout_FEEL'] + list_var_dummy #+ list_var_apres_encodage

# CENTRAGE ET RÉDUCTION #
X = df[var_expl_to_test]
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(df[var_expl_to_test]), columns=var_expl_to_test)


#################################################################
#                    ANALYSE UNIVARIÉE                          #
#################################################################

dict_stats_attrition_variable = {}
for var in var_expl_to_test :
    dict_stats_attrition_variable[var] = df.groupby(VarToPredict)[var].describe()
df_stats_attrition_variable = pd.concat(dict_stats_attrition_variable.values(), keys=dict_stats_attrition_variable.keys())

# ZOOM : NOMBRE D'ADJECTIFS #
df['nb_adjectifs_cat'] = df['nb_adjectifs'].apply(lambda x: '0' if x == 0 else '1' if x == 1 else '>1')
df.groupby('nb_adjectifs_cat')[VarToPredict+'_num'].mean()

# ZOOM : NOMBRE DE POINTS D'EXCLAMATION #
df['nbr_exclam_cat'] = df['nbr_exclam'].apply(lambda x: '0' if x == 0 else '1' if x == 1 else '>1')
df.groupby('nbr_exclam_cat')[VarToPredict+'_num'].mean()

# ZOOM : NOMBRE DE MOTS #
df.groupby('nombre_mots')[VarToPredict+'_num'].mean()

X = df[["nombre_mots"]]     # variable à discrétiser
y = df[VarToPredict+'_num'] # variable cible
clf = DecisionTreeClassifier(max_depth=3) 
clf.fit(X, y)
fig, ax = plt.subplots(figsize=(10, 6))
plot_tree(clf, filled=True, ax=ax, feature_names=["nombre de mots"])
plt.show()

df['nombre_mots_cat'] = df['nombre_mots'].apply(lambda x: '<8' if x < 8 else '1' if x == 1 else '>1')


# ZOOM : NOMBRE DE MOTS POSITIFS SELON LE LEXIQUE FEEL #
df.groupby('nombre_mots_positifs_FEEL')[VarToPredict+'_num'].mean()



#################################################################
#                    ANALYSE SPATIALE                           #
#################################################################










#################################################################
#                   ANALYSE MULTIVARIÉE                         #
#################################################################




# ÉCHANTILLONS #
#X_train, X_test, y_train, y_test = train_test_split(df[var_expl_to_test], df[VarToPredict+'_num'], test_size=0.2, random_state=99)
X_train, X_test, y_train, y_test = train_test_split(X, df[VarToPredict+'_num'], test_size=0.2, random_state=99)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df[VarToPredict+'_num'], test_size=0.2, random_state=99)



# SANS RÉÉQUILIBRAGE #
######################
    # REGRESSION LOGISTIQUE #
    #########################
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_proba = lr.predict_proba(X_test)
auc = roc_auc_score(y_test, y_proba[:, 1])
print("AUC : ",auc)

for threshold in thresholds:
    y_proba = lr.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))
    f1 = f1_score(y_test, y_pred)
    print("F1 score : ", f1)
    

    # Coefficients du modèle #
constante = lr.intercept_[0]
print("constante :", constante)
coef = lr.coef_
variable_names = pd.DataFrame(df[var_expl_to_test]).columns
print("coefficients :", coef)
for i in range(len(variable_names)):
    print(variable_names[i], ":", coef[0][i])

    # Régression logistique sans constante #
    ########################################
lr_no_intercept = LogisticRegression(fit_intercept=False)
lr_no_intercept.fit(X_train, y_train)
y_proba = lr_no_intercept.predict_proba(X_test)
auc = roc_auc_score(y_test, y_proba[:, 1])
print("AUC : ",auc)

    # Coefficients du modèle #
constante = lr_no_intercept.intercept_[0]
print("constante :", constante)
coef = lr_no_intercept.coef_
variable_names = pd.DataFrame(df[var_expl_to_test]).columns
print("coefficients :", coef)
for i in range(len(variable_names)):
    print(variable_names[i], ":", coef[0][i])

for threshold in thresholds:
    y_proba = lr_no_intercept.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))





    # GRADIANT BOOSTING #
    #####################
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_proba = gb.predict_proba(X_test)
auc = roc_auc_score(y_test, y_proba[:, 1])
print("AUC : ",auc)

for threshold in thresholds:
    y_proba = gb.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))
    
    

    # XG-BOOST #
    ############
xgb = XGBClassifier(max_depth=12, learning_rate=0.01, n_estimators=400, \
                    objective='binary:logistic', n_jobs = 8, \
                    gamma=0, min_child_weight=5, 
                    max_delta_step=0, subsample=0.75, colsample_bytree=0.75,
                    scale_pos_weight= 1, missing=None)
xgb.fit(X_train, y_train)
y_proba = xgb.predict_proba(X_test)[:,1]
auc_xgboost = roc_auc_score(y_test, y_proba)
print("AUC : ",auc_xgboost)

for threshold in thresholds:
    y_proba = xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm_xgboost = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))
    
   

    # => K PLUS PROCHES VOISINS #
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_proba = knn.predict_proba(X_test)
auc_knn = roc_auc_score(y_test, y_proba[:, 1])
print(f"AUC: {auc_knn}")   

for threshold in thresholds:
    y_proba = knn.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))
    
    



# AVEC RANDOM OVER SAMPLER #
############################
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)



    # => RÉGRESSION LOGISTIQUE #
lr = LogisticRegression(max_iter = 150)
lr.fit(X_train_ros, y_train_ros)
y_proba = lr.predict_proba(X_test)
auc = roc_auc_score(y_test, y_proba[:, 1])
print("AUC : ",auc)

for threshold in thresholds:
    y_proba = lr.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))
    f1 = f1_score(y_test, y_pred)
    print("F1 score : ", f1)

    # Coefficients du modèle #
constante = lr.intercept_[0]
print("constante :", constante)
coef = lr.coef_
variable_names = pd.DataFrame(df[var_expl_to_test]).columns
print("coefficients :", coef)
for i in range(len(variable_names)):
    #print(variable_names[i], ":", coef[0][i])
    print(variable_names[i], ":", np.exp(coef[0][i])) # calcul du rapport de cotes #
    
  
    
    # => RÉGRESSION LOGISTIQUE PÉNALISÉE l1 LASSO #
lr_lasso_ros = LogisticRegression(penalty='l1', solver='liblinear')
lr_lasso_ros.fit(X_train_ros, y_train_ros)
y_proba = lr_lasso_ros.predict_proba(X_test)
auc_lr_lasso_ros = roc_auc_score(y_test, y_proba[:, 1])
print("AUC : ",auc_lr_lasso_ros)
    
for threshold in thresholds:
    y_proba = lr_lasso_ros.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))

    # Coefficients du modèle #
constante = lr_lasso_ros.intercept_[0]
print("constante :", constante)
coef = lr_lasso_ros.coef_
variable_names = pd.DataFrame(df[var_expl_to_test]).columns
print("coefficients :", coef)
for i in range(len(variable_names)):
    print(variable_names[i], ":", coef[0][i])



    # => RÉGRESSION LOGISTIQUE PÉNALISÉE l2 RIDGE #
lr_ridge_ros = LogisticRegression(penalty='l2')
lr_ridge_ros.fit(X_train_ros, y_train_ros)
y_proba = lr_ridge_ros.predict_proba(X_test)
auc_lr_ridge_ros = roc_auc_score(y_test, y_proba[:, 1])
print("AUC : ",auc_lr_ridge_ros)
    
for threshold in thresholds:
    y_proba = lr_ridge_ros.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))



    # => RÉGRESSION LOGISTIQUE SIMPLE SANS CONSTANTE #
lr_no_intercept_ros = LogisticRegression(fit_intercept=False)
lr_no_intercept_ros.fit(X_train_ros, y_train_ros)
y_proba = lr_no_intercept_ros.predict_proba(X_test)
auc = roc_auc_score(y_test, y_proba[:, 1])
print("AUC : ",auc)
for threshold in thresholds:
    y_proba = lr_no_intercept_ros.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))



    # => GRADIANT BOOSTING #
gb = GradientBoostingClassifier()
gb.fit(X_train_ros, y_train_ros)
y_proba = gb.predict_proba(X_test)
auc = roc_auc_score(y_test, y_proba[:, 1])
print("AUC : ",auc)

y_proba = gb.predict_proba(X_test)[:, 1]
for threshold in thresholds:
    y_proba = gb.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))
    
    # => XG-BOOST #
xgb = XGBClassifier(max_depth=12, learning_rate=0.01, n_estimators=400, \
                    objective='binary:logistic', n_jobs = 8, \
                    gamma=0, min_child_weight=5, 
                    max_delta_step=0, subsample=0.75, colsample_bytree=0.75,
                    scale_pos_weight= 1, missing=None)
xgb.fit(X_train_ros,y_train_ros)
y_proba = xgb.predict_proba(X_test)[:,1]
auc_ros_xgboost = roc_auc_score(y_test, y_proba)
print(f"AUC: {auc_ros_xgboost}")

importance_scores = xgb.get_booster().get_score(importance_type='weight')
importance_var_xgboost_ros = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['importance'])
importance_var_xgboost_ros = importance_var_xgboost_ros.sort_values('importance', ascending=False)

y_proba = xgb.predict_proba(X_test)[:, 1]
for threshold in thresholds:
    y_proba = xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))


    # => RF #
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500) # 500 arbres de décisions #
rf.fit(X_train_ros,y_train_ros)
y_proba = rf.predict_proba(X_test)
auc_ros_rf = roc_auc_score(y_test, y_proba[:, 1])
print(f"AUC: {auc_ros_rf}")


    # => SVM #
svm_ros = SVC(kernel='linear', probability=True)
svm_ros.fit(X_train_ros, y_train_ros)  
y_proba = svm_ros.predict_proba(X_test)
auc_svm_ros = roc_auc_score(y_test, y_proba[:, 1])
print(f"AUC: {auc_svm_ros}")   
  

    # => K PLUS PROCHES VOISINS #
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_ros, y_train_ros)
y_proba = knn.predict_proba(X_test)
auc_ros_knn = roc_auc_score(y_test, y_proba[:, 1])
print(f"AUC: {auc_ros_knn}")   

for threshold in thresholds:
    y_proba = knn.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))

 
    
    
  

# AVEC SMOTE #
##############
sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)


    # => RÉGRESSION LOGISTIQUE #
lr_smote = LogisticRegression()
lr_smote.fit(X_train_smote, y_train_smote)
y_proba = lr_smote.predict_proba(X_test)
auc = roc_auc_score(y_test, y_proba[:, 1])
print("AUC : ",auc)

for threshold in thresholds:
    y_proba = lr_smote.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))

    # Coefficients du modèle #
constante = lr_smote.intercept_[0]
print("constante :", constante)
coef = lr_smote.coef_
variable_names = pd.DataFrame(df[var_expl_to_test]).columns
print("coefficients :", coef)
for i in range(len(variable_names)):
    print(variable_names[i], ":", coef[0][i])
    
    
    
    
    

    # => GRADIANT BOOSTING #
gb = GradientBoostingClassifier()
gb.fit(X_train_smote, y_train_smote)
y_proba = gb.predict_proba(X_test)
auc = roc_auc_score(y_test, y_proba[:, 1])
print("AUC : ",auc)

y_proba = gb.predict_proba(X_test)[:, 1]
for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))
    
importances = gb.feature_importances_
importance_var_gb_smote = pd.DataFrame({'feature': X_train_smote.columns, 'importance': importances})
importance_var_gb_smote = importance_var_gb_smote.sort_values('importance', ascending=False).reset_index(drop=True)


    
    # => XG-BOOST #
xgb = XGBClassifier(max_depth=12, learning_rate=0.01, n_estimators=400, \
                    objective='binary:logistic', n_jobs = 8, \
                    gamma=0, min_child_weight=5, 
                    max_delta_step=0, subsample=0.75, colsample_bytree=0.75,
                    scale_pos_weight= 1, missing=None)
xgb.fit(X_train_smote,y_train_smote)
y_proba = xgb.predict_proba(X_test)[:,1]
auc_xgboost_smote = roc_auc_score(y_test, y_proba)
print(f"AUC: {auc_xgboost_smote}")

importance_scores = xgb.get_booster().get_score(importance_type='weight')
importance_var_xgboost_smote = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['importance'])
importance_var_xgboost_smote = importance_var_xgboost_smote.sort_values('importance', ascending=False)

y_proba = xgb.predict_proba(X_test)[:, 1]
for threshold in thresholds:
    y_proba = xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))
    
    
    
    # => K PLUS PROCHES VOISINS #
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_smote, y_train_smote)
y_proba = knn.predict_proba(X_test)
auc_smote_knn = roc_auc_score(y_test, y_proba[:, 1])
print(f"AUC: {auc_smote_knn}")   

for threshold in thresholds:
    y_proba = knn.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f" ********** seuil {threshold} **********")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)
    print(classification_report(y_test, y_pred))