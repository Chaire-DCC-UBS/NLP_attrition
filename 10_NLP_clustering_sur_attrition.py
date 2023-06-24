#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:44:36 2023

@author: ac
"""
###############################################################################
#                  CLUSTERING DES ATTRITIONS SEULES                           #
###############################################################################

import hdbscan
from sklearn.cluster import DBSCAN

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
df_w2v_attr = pd.DataFrame(np.column_stack((coord, top_attrition)), columns=['x', 'y', 'top_attrition'])


#####################################################
# CLUSTERING UNIQUEMENT SUR LES POINTS EN ATTRITION #
#####################################################

df_w2v_attr_1 = df_w2v_attr.loc[df_w2v_attr['top_attrition']==1]
df_w2v_attr_1.drop(['top_attrition'], axis=1, inplace=True)

# AVEC K-MEANS #
kmeans = KMeans(n_clusters=4, random_state=99,init='k-means++').fit(df_w2v_attr_1)
cluster_labels = kmeans.predict(df_w2v_attr_1)

fig, ax = plt.subplots(figsize=(10,10))
scatter = ax.scatter(df_w2v_attr_1['x'], df_w2v_attr_1['y'], c=cluster_labels, cmap='rainbow')
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
ax.add_artist(legend)
plt.show()

# AVEC HDBSCAN, POUR PRENDRE EN COMPTE LES DIFFÉRENCES DE DENSITÉ #
hdbscan = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=2)
clusters_attr_hdbscan = hdbscan.fit_predict(df_w2v_attr_1)
df_w2v_attr_1['clusters_attr_hdbscan'] = clusters_attr_hdbscan

fig, ax = plt.subplots(figsize=(10,10))
scatter = ax.scatter(df_w2v_attr_1['x'], df_w2v_attr_1['y'], c=clusters_attr_hdbscan, cmap='rainbow')
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
ax.add_artist(legend)
plt.show()

# INTÉGRATION DU CLUSTER ATTRITION ISSU DE HDBSCAN DANS LA BASE INITIALE #
comments = pd.merge(comments, df_w2v_attr_1[['clusters_attr_hdbscan']], left_index=True, right_index=True, how='left')

# ANALYSE DES COMMENTAIRES SELON LE CLUSTER D'ATTRITION #
clusters_attr_hdbscan_list = comments['clusters_attr_hdbscan'].dropna().unique()
for cluster in clusters_attr_hdbscan_list :
    print('cluster attrition hdbscan : '+ str(cluster))
    cluster_df = comments[comments['clusters_attr_hdbscan'] == cluster]
    words_list = [word for words in cluster_df['var_lemma_nosw'] for word in words]
    word_freq = Counter(words_list)   
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(12,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# ANALYSE DE LA NOTE DE RECOMMANDATION DONNÉE PAR CLUSTER D'ATTRITION #
for cluster in clusters_attr_hdbscan_list :  
    cluster_df = comments[comments['clusters_attr_hdbscan'] == cluster]
    print('cluster attrition hdbscan : '+ str(cluster) )
    print('effectif : ' + str(len(cluster_df)))
    print('moyenne note de recommandation : '+ str(cluster_df['Note_recommandation'].mean()))
    



