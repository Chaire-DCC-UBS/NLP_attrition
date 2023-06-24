#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:06:06 2023

@author: ac
"""



### CODE A LANCER PAS À PAS, VOIRE EN ALLER-RETOUR, POUR AJUSTER LES DIFFÉRENTS HYPERPARAMÈTRES EN FONCTION DES RÉSUTATS ###






from gensim.models import word2vec
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

comments = comments.reset_index()

#  HYPERPARAMÈTRES #
####################

architecture_w2v = 0     # 0 : CBOW, 1 : SKIP-GRAM #
size_w2v = 5             # dimension vecteur w2v #
window_w2v = 10          # fenêtre avant / après le mot #
perplexity = 100         # perplexité TSNE



from gensim.models.phrases import Phrases, Phraser
sentences = [comment.split() for comment in comments['var_lemma_nosw_txt']]
bigram_model = Phrases(sentences, min_count=10, threshold=10)
bigram_phraser = Phraser(bigram_model)
bigram_sentences = [bigram_phraser[comment] for comment in sentences]
comments['bigram_sentences'] = bigram_sentences

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

# Avec prise en compte des ngrams #
modelw2v_bigram = word2vec.Word2Vec(bigram_sentences,
                          min_count=5,
                          size=size_w2v,
                          workers=1, 
                          sg=architecture_w2v,
                          window=window_w2v,     
                          seed = 99,
                          iter=100)

vocab_w2v_unigram   = list(modelw2v_unigram.wv.vocab.keys())
vocab_w2v_bigram    = list(modelw2v_bigram.wv.vocab.keys())




#  VECTORISATION DES PROPOSITIONS  #
####################################
# MOYENNE SIMPLE #
def word_averaging(modelwv, words):
    words_to_mean=[]
    for word in words:
        if word in modelwv.wv.vocab :
            words_to_mean.append(modelwv.wv[word])  
    return(np.array(words_to_mean).mean(axis=0)) 
comments['w2v_vector_mean_unigram']=comments['var_lemma_nosw'].apply(lambda x:word_averaging(modelw2v_unigram,x))
comments['w2v_vector_mean_bigram'] = comments['bigram_sentences'].apply(lambda x: word_averaging(modelw2v_bigram, x))

# Affectation d'un vecteur nul aux propositions vides 0
zero_vector = np.zeros((size_w2v,), dtype="float32")
comments["w2v_vector_mean_unigram"] = comments.apply(lambda row: zero_vector if row["var_lemma_nosw_txt"] == "" else row["w2v_vector_mean_unigram"], axis=1)

comments["w2v_vector_mean_bigram"] = comments.apply(lambda row: zero_vector if row["bigram_sentences"] == "" else row["w2v_vector_mean_bigram"], axis=1)

# avec unigrams #
idx_na_mean_unigram = comments['w2v_vector_mean_unigram'].isna()
tab_mean_unigram = pd.DataFrame(data=comments.loc[~idx_na_mean_unigram, 'w2v_vector_mean_unigram'].tolist(), index=comments.loc[~idx_na_mean_unigram, 'w2v_vector_mean_unigram'].index)
df_prop_vect_w2v_mean_unigram = pd.concat([comments['var_lemma_nosw'],tab_mean_unigram], axis=1)

# avec bigrams #
idx_na_mean_bigram = comments['w2v_vector_mean_bigram'].isna()
tab_mean_bigram = pd.DataFrame(data=comments.loc[~idx_na_mean_bigram, 'w2v_vector_mean_bigram'].tolist(), index=comments.loc[~idx_na_mean_bigram, 'w2v_vector_mean_bigram'].index)
df_prop_vect_w2v_mean_bigram = pd.concat([comments['bigram_sentences'],tab_mean_bigram], axis=1)



# MOYENNE AVEC PONDÉRATION TF-IDF #
tfidf = TfidfVectorizer()
tfidf.fit_transform(comments['var_lemma_nosw_txt'])
feature_names = tfidf.get_feature_names_out()
word2tfidf = dict(zip(feature_names, tfidf.idf_))

def word_averaging_tfidf(model, words, word2tfidf):
    all_words, mean = set(), []
    for word in words:
        if word in model.wv.vocab and word in word2tfidf:
            mean.append(model.wv[word] * word2tfidf[word])
            all_words.add(word)
    if not mean:
        return np.zeros(model.vector_size,)
    mean = np.array(mean).mean(axis=0)
    return mean

# avec unigrams #
comments['w2v_vector_meantfidf_unigram'] = [word_averaging_tfidf(modelw2v_unigram, comment.split(), word2tfidf) for comment in comments['var_lemma_nosw_txt']]   
# Affectation d'un vecteur nul aux propositions vides 0
comments["w2v_vector_meantfidf_unigram"] = comments.apply(lambda row: zero_vector if row["var_lemma_nosw_txt"] == "" else row["w2v_vector_meantfidf_unigram"], axis=1)

idx_na_meantfidf_unigram = comments['w2v_vector_meantfidf_unigram'].isna()
tab_meantfidf_unigram = pd.DataFrame(data=comments.loc[~idx_na_meantfidf_unigram, 'w2v_vector_meantfidf_unigram'].tolist(), index=comments.loc[~idx_na_meantfidf_unigram, 'w2v_vector_meantfidf_unigram'].index)
df_prop_vect_w2v_meantfidf_unigram = pd.concat([comments['var_lemma_nosw'],tab_meantfidf_unigram], axis=1)


# avec bigrams #
comments['w2v_vector_meantfidf_bigram'] = [word_averaging_tfidf(modelw2v_bigram, comment, word2tfidf) for comment in comments['bigram_sentences']]   
# Affectation d'un vecteur nul aux propositions vides 0
comments["w2v_vector_meantfidf_bigram"] = comments.apply(lambda row: zero_vector if row["bigram_sentences"] == "" else row["w2v_vector_meantfidf_bigram"], axis=1)

idx_na_meantfidf_bigram = comments['w2v_vector_meantfidf_bigram'].isna()
tab_meantfidf_bigram = pd.DataFrame(data=comments.loc[~idx_na_meantfidf_bigram, 'w2v_vector_meantfidf_bigram'].tolist(), index=comments.loc[~idx_na_meantfidf_bigram, 'w2v_vector_meantfidf_bigram'].index)
df_prop_vect_w2v_meantfidf_bigram = pd.concat([comments['bigram_sentences'],tab_meantfidf_bigram], axis=1)
    
     
#  RÉDUCTION DE DIMENSION  #
############################
tsne = TSNE(n_components=2,  perplexity=perplexity, random_state=99)
tab_mean_unigram_tsne       = tsne.fit_transform(tab_mean_unigram)
tab_mean_bigram_tsne        = tsne.fit_transform(tab_mean_bigram)
tab_mean_unigram_tfidf_tsne = tsne.fit_transform(tab_meantfidf_unigram)
tab_mean_bigram_tfidf_tsne  = tsne.fit_transform(tab_meantfidf_bigram)

#  VISUALISATION APRES RÉDUCTION DE DIMENSION #
###############################################
plt.scatter(tab_mean_unigram_tsne[:,0], tab_mean_unigram_tsne[:,1])
plt.show()

plt.scatter(tab_mean_bigram_tsne[:,0], tab_mean_bigram_tsne[:,1])
plt.show()

plt.scatter(tab_mean_unigram_tfidf_tsne[:,0], tab_mean_unigram_tfidf_tsne[:,1])
plt.show()

plt.scatter(tab_mean_bigram_tfidf_tsne[:,0], tab_mean_bigram_tfidf_tsne[:,1])
plt.show()


#  CLUSTERING #
###############

# !!! CHOIX DE LA MOYENNE UTILISÉE A FAIRE ICI !!! #
#mytab = tab_mean_unigram_tfidf_tsne
mytab = tab_mean_unigram_tsne

# K-MEANS - INITIALISATION RANDOM #
# CHOIX DU MEILLEUR NOMBRE DE CLUSTERS EN FONCTION DU COEFFICIENT DE SILHOUETTE #
nb_clusters_list = range(5, 10)
silhouette_scores = []
for n_clusters in nb_clusters_list:
    kmeans = KMeans(n_clusters=n_clusters, random_state=99,)
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

# K-MEANS - INITIALISATION K-MEANS++ #
# CHOIX DU MEILLEUR NOMBRE DE CLUSTERS EN FONCTION DU COEFFICIENT DE SILHOUETTE #
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
nb_clusters = 7
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


# POINTS LES PLUS PROCHES DE CHAQUE CENTROÏDES #
distances = cdist(mytab, centroids)
closest_indices = np.argmin(distances, axis=0)
clustered_comments = comments.iloc[closest_indices]


# AJOUT DES LABELS DANS LA BASE INITIALE #
if 'label_cluster' in comments.columns:
    comments['label_cluster'] = labels
else:
    comments = pd.concat([comments, pd.DataFrame(labels, columns=['label_cluster'])], axis=1)



# INTERPRETATION DES CLUSTERS #
unique_labels = comments['label_cluster'].unique()

# avec unigrams #
for label in unique_labels:
    print('cluster : '+ str(label))
    cluster_df = comments[comments['label_cluster'] == label]
    words_list = [word for words in cluster_df['var_lemma_nosw'] for word in words]
    word_freq = Counter(words_list)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(12,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# avec bigrams #
for label in unique_labels:
    print('cluster : '+ str(label))
    cluster_df = comments[comments['label_cluster'] == label]
    words_list = [word for words in cluster_df['bigram_sentences'] for word in words]
    word_freq = Counter(words_list)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(12,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

##################### SAUVEGARDE DE LA BASE OBTENUE ###########################
comments.to_csv('comments_20230503.csv')

##################### IMPORT D'UNE BASE SAUVEGARDÉE ###########################
comments = pd.read_csv('comments_20230503.csv')
exec(open('NLP_preprocessing.py').read()) # pour bien récupérer les variables au bon format #

