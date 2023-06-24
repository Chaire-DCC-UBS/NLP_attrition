
# ANALYSE DE L'ATTRITION PAR CLUSTERS #

VarToPredict = 'ATTR_CONC'
list_modalites_y = ['TOT','SANS']
top_attr = 'TOT'
top_no_attr = 'SANS'

VarToPredict = 'ATTR_CONC_TOTPART_SANS'
list_modalites_y = ['TOT_OU_PART','SANS']
top_attr = 'TOT_OU_PART'
top_no_attr = 'SANS'


df_attr = comments.loc[comments[VarToPredict].isin(list_modalites_y)]
df_attr[VarToPredict+'_num'] = df_attr[VarToPredict].map({top_no_attr: 0, top_attr: 1})

# TAUX D'ATTRITION TOTAL #
tx_attr = df_attr[VarToPredict+'_num'].mean()

# TAUX D'ATTRITION PAR CLUSTERS #
tx_attr_by_cluster = df_attr.groupby('label_cluster')[VarToPredict+'_num'].mean()

# EFFECTIF ATTRITION / NON ATTRITION PAR CLUSTER #
ct = pd.crosstab(df_attr['label_cluster'], df_attr[VarToPredict+'_num'])

# GRAPHE DES POINTS : ROUGE SI ATTRITION, BLEU SI NON ATTRITION #
colors = np.where(comments[VarToPredict] == 'TOT', 'r', 'b')
sizes = np.where(comments[VarToPredict] == 'TOT', 15, 10)
plt.scatter(mytab[:,0], mytab[:,1], c=colors, s=sizes)
plt.show()

