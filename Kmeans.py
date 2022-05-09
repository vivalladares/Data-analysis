'''------------KMeans------------'''

import pandas as pd 
import matplotlib.pyplot as plt

base = pd.read_csv('frios3.csv') 
base.describe() 

#Pre-processamento

base_remove = base.loc[(base['psuc10'] <2.10 )]
base = base.drop(base_remove.index)

base_remove1 = base.loc[(base['psuc10'] > 3 )]
base = base.drop(base_remove1.index)

base_remove2 = base.loc[(base['cap10'] == 0 )]
base = base.drop(base_remove2.index)

#Redução de dimensionalidade StandardScaler

from sklearn.preprocessing import StandardScaler


X = base.iloc[:,0:6].values
scaler=StandardScaler()
X= scaler.fit_transform(X)

#Método de Elbow, número de clusters recomendado

from sklearn.cluster import KMeans

wcss = []
 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state=(0))
    kmeans.fit(X)
    print (i)
    print(kmeans.inertia_)
    wcss.append(kmeans.inertia_) 


plt.plot(range(1, 11), wcss)
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()


#K-Means, K = 3

kmeans = KMeans(n_clusters=3, random_state = 0)
previsoes = kmeans.fit_predict(X)

#Plot Clusters

plt.scatter(X[previsoes == 0,2], X[previsoes == 0, 5], s=100, c='red', label = 'Cluster 1' )
plt.scatter(X[previsoes == 1,2], X[previsoes == 1, 5], s=100, c='green' , label = 'Cluster 2')
plt.scatter(X[previsoes == 2,2], X[previsoes == 2, 5], s=100, c='blue' , label = 'Cluster 3')
plt.xlabel('psuc')
plt.ylabel('cap')
plt.legend()



