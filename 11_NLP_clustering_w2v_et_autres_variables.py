#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 21:40:07 2023

@author: ac
"""

###############################################################################
#                         MODÉLISATION SPATIALE                               #
###############################################################################

from gensim.models import word2vec
import hdbscan
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr



#  HYPERPARAMÈTRES #
####################

architecture_w2v = 0     # 0 : CBOW, 1 : SKIP-GRAM #
size_w2v = 5             # dimension vecteur w2v #
window_w2v = 10          # fenêtre avant / après le mot #
perplexity = 100         # perplexité TSNE
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


#  VECTORISATION DES MOTS  #
############################
# Sans prendre en compte les ngrams #
modelw2v_unigram = word2vec.Word2Vec(comments['var_lemma_nosw'],
                          min_count=1,
                          size=size_w2v,
                          workers=1, 
                          sg=architecture_w2v,
                          window=window_w2v,     
                          seed = 99,
                          iter=100)


#  VECTORISATION DES PROPOSITIONS  #
####################################
# MOYENNE SIMPLE #
def word_averaging(modelwv, words):
    words_to_mean=[]
    for word in words:
        if word in modelwv.wv.vocab :
            words_to_mean.append(modelwv.wv[word])  
    return(np.array(words_to_mean).mean(axis=0)) 
comments['w2v_results']=comments['var_lemma_nosw'].apply(lambda x:word_averaging(modelw2v_unigram,x))

# Affectation d'un vecteur nul aux propositions vides 0
zero_vector = np.zeros((size_w2v,), dtype="float32")
comments["w2v_results"] = comments.apply(lambda row: zero_vector if row["var_lemma_nosw_txt"] == "" else row["w2v_results"], axis=1)

# avec unigrams #
idx_na_mean_unigram = comments['w2v_results'].isna()
tab_mean_unigram = pd.DataFrame(data=comments.loc[~idx_na_mean_unigram, 'w2v_results'].tolist(), index=comments.loc[~idx_na_mean_unigram, 'w2v_results'].index)
df_prop_vect_w2v_mean_unigram = pd.concat([comments['var_lemma_nosw'],tab_mean_unigram], axis=1)


#  FUSION AVEC LES AUTRES VARIABLES  #
######################################
var_expl_to_test_sans_clusters = [
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
'nombre_mots_degout_FEEL'] 

tab_w2v_var = pd.merge(tab_mean_unigram, comments[var_expl_to_test_sans_clusters], left_index =True , right_index= True)
tab_w2v_var.columns = tab_w2v_var.columns.astype(str)
tsne = TSNE(n_components=2,  perplexity=perplexity, random_state=99)
tab_w2v_var_tsne = tsne.fit_transform(tab_w2v_var)



#  VISUALISATION APRES RÉDUCTION DE DIMENSION #
###############################################
plt.scatter(tab_w2v_var_tsne[:,0], tab_w2v_var_tsne[:,1])
plt.show()



#  CLUSTERING #
###############
mytab = tab_w2v_var_tsne

nb_clusters_list = range(5, 10)
silhouette_scores = []
for n_clusters in nb_clusters_list:
    kmeans = KMeans(n_clusters=n_clusters, random_state=99, init='k-means++')
    kmeans.fit(mytab)
    labels = kmeans.predict(mytab)
    score = silhouette_score(mytab, labels)
    silhouette_scores.append(score)

plt.plot(nb_clusters_list, silhouette_scores)
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de silhouette')
plt.show()
best_nb_clusters = np.argmax(silhouette_scores) + 5 # +5 car on commence avec 5 clusters
print('Le nombre optimal de clusters est :', best_nb_clusters)

# K-MEANS FINALEMENT RETENU #
nb_clusters = 50
kmeans = KMeans(n_clusters=nb_clusters, random_state=99, init='k-means++')
kmeans.fit(mytab)
labels = kmeans.predict(mytab)
centroids = kmeans.cluster_centers_

# VISUALISATION DES CLUSTERS #
fig, ax = plt.subplots(figsize=(10,10))
scatter = ax.scatter(mytab[:,0], mytab[:,1], c=labels, cmap='rainbow')
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
ax.add_artist(legend)
plt.show()


# AJOUT DES LABELS DANS LA BASE INITIALE #
if 'label_cluster_w2v_et_var' in comments.columns:
    comments['label_cluster_w2v_et_var'] = labels
else:
    comments = pd.concat([comments, pd.DataFrame(labels, columns=['label_cluster_w2v_et_var'])], axis=1)
    

# ANALYSE DE L'ATTRITION PAR CLUSTER #

VarToPredict = 'ATTR_CONC'
list_modalites_y = ['TOT','SANS']
top_attr = 'TOT'
top_no_attr = 'SANS'

df_attr = comments.loc[comments[VarToPredict].isin(list_modalites_y)]
df_attr[VarToPredict+'_num'] = df_attr[VarToPredict].map({top_no_attr: 0, top_attr: 1})

# TAUX D'ATTRITION TOTAL #
tx_attr = df_attr[VarToPredict+'_num'].mean()

# TAUX D'ATTRITION PAR CLUSTERS #
tx_attr_by_cluster = df_attr.groupby('label_cluster_w2v_et_var')[VarToPredict+'_num'].mean()

# EFFECTIF ATTRITION / NON ATTRITION PAR CLUSTER #
ct = pd.crosstab(df_attr['label_cluster_w2v_et_var'], df_attr[VarToPredict+'_num'])

# GRAPHE DES POINTS : ROUGE SI ATTRITION, BLEU SI NON ATTRITION #
colors = np.where(comments[VarToPredict] == 'TOT', 'r', 'b')
sizes = np.where(comments[VarToPredict] == 'TOT', 15, 10)
plt.scatter(mytab[:,0], mytab[:,1], c=colors, s=sizes)
plt.show()


###############################################################################
#                  CLUSTERING DES ATTRITIONS SEULES                           #
###############################################################################

# DÉFINITION DE LA VARIABLE D'ATTRITION #
VarToPredict = 'ATTR_CONC'

def attrition_to_binary_tot_other(attrition):
    if attrition == "TOT":
        return 1
    else:
        return 0
comments["attrition_tot_vs_other"] = comments[VarToPredict].apply(attrition_to_binary_tot_other)

# Calcul du taux d'attrition global #
attr_eff = comments['attrition_tot_vs_other'].value_counts()
attr_pct = comments['attrition_tot_vs_other'].value_counts(normalize=True)

# AGRÉGATION DES COORDONNÉES ET DES DONNÉES D'ATTRITION # 
top_attrition = comments['attrition_tot_vs_other']
coord = np.array(mytab)
df_w2v_var_attr = pd.DataFrame(np.column_stack((coord, top_attrition)), columns=['x', 'y', 'top_attrition'])


#####################################################
# CLUSTERING UNIQUEMENT SUR LES POINTS EN ATTRITION #
#####################################################

df_w2v_var_attr_1 = df_w2v_var_attr.loc[df_w2v_var_attr['top_attrition']==1]
df_w2v_var_attr_1.drop(['top_attrition'], axis=1, inplace=True)

# AVEC K-MEANS #
kmeans = KMeans(n_clusters=4, random_state=99,init='k-means++').fit(df_w2v_var_attr_1)
cluster_labels = kmeans.predict(df_w2v_var_attr_1)

fig, ax = plt.subplots(figsize=(10,10))
scatter = ax.scatter(df_w2v_var_attr_1['x'], df_w2v_var_attr_1['y'], c=cluster_labels, cmap='rainbow')
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
ax.add_artist(legend)
plt.show()

# AVEC HDBSCAN, POUR PRENDRE EN COMPTE LES DIFFÉRENCES DE DENSITÉ #
hdbscan = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=2)
clusters_attr_hdbscan = hdbscan.fit_predict(df_w2v_var_attr_1)
df_w2v_var_attr_1['clusters_attr_hdbscan'] = clusters_attr_hdbscan

fig, ax = plt.subplots(figsize=(10,10))
scatter = ax.scatter(df_w2v_var_attr_1['x'], df_w2v_var_attr_1['y'], c=clusters_attr_hdbscan, cmap='rainbow')
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
ax.add_artist(legend)
plt.show()


#####################################################
#           INTERPRETATION DES AXES                 #
#####################################################

# Calcul du coeff de Pearson entre variables de bases et chaque dim de TSNE #
tab_w2v_var = tab_w2v_var.astype(float)
loadings = []
for i in range(tab_w2v_var.shape[1]):
    corr, pval = pearsonr(tab_w2v_var.iloc[:,i], tab_w2v_var_tsne[:,0])
    loadings.append(corr)
    
    corr, pval = pearsonr(tab_w2v_var.iloc[:,i], tab_w2v_var_tsne[:,1])
    loadings.append(corr)
    
loadings = pd.DataFrame(np.array(loadings).reshape((tab_w2v_var.shape[1], 2)), columns=['tsne_1', 'tsne_2'], index=tab_w2v_var.columns)

plt.figure(figsize=(10, 10))
plt.scatter(loadings.iloc[:, 0], loadings.iloc[:, 1], s=50, alpha=0.8)
for i, txt in enumerate(tab_w2v_var.index):
    plt.annotate(txt, (loadings.iloc[i, 0], loadings.iloc[i, 1]), fontsize=10)
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.title('Variable loadings in t-SNE space')

    
    
    

#######################################################################
# TESTS DE MODÉLISATION PAR K-PLUS PROCHES VOISINS SUR TABLEAU RÉDUIT #
#######################################################################
# CENTRAGE ET RÉDUCTION #
X = df_w2v_var_attr[['x','y']]
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, df_w2v_var_attr['top_attrition'], test_size=0.2, random_state=99)


# AVEC RANDOM OVER SAMPLER #
############################
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

    # => K PLUS PROCHES VOISINS #
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

