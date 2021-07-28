import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.activations import relu, sigmoid
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from keras.layers import Dense, Input

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# #数据概况
raw_file = pd.read_csv("./toy_GCs.csv")
df = pd.DataFrame(raw_file)
data1 = df.drop('GCs', 1)

# 数据标准化
features = ['u_hat', 'j378_hat', 'j395_hat', 'j410_hat', 'j430_hat', 'g_hat',
            'j515_hat', 'r_hat', 'j660_hat', 'i_hat', 'j861_hat', 'z_hat']
x = df.loc[:, features].values
y = df.loc[:, ['GCs']].values
x = StandardScaler().fit_transform(x)
print(pd.DataFrame(data=x, columns=features).head())

# 检查方差
pca = PCA(n_components=12)
principalComponents = pca.fit_transform(x)
print(pca.explained_variance_ratio_)

# 方差相关度制图
importance = pca.explained_variance_ratio_
plt.scatter(range(1, 13), importance)
plt.plot(range(1, 13), importance)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# PCA降维
pca = PCA(n_components=0.99)
principalComponents = pca.fit_transform(x)
# 查看降维后的数据
principalDf = pd.DataFrame(data=principalComponents)
finalDf = pd.concat([principalDf, df[['GCs']]], axis=1)
print(finalDf.head(5))
finalDf.to_csv("./pca.csv")
labels = finalDf[finalDf['GCs'] == 1]
unlabels = finalDf[finalDf['GCs'] == 0]
unlabels.to_csv('./unlabel.csv')
labels.to_csv("./label.csv")

# 纯K-means算法
# inertia = []
# for k in range(1,10):
#     kmodel = KMeans(n_clusters=k,n_jobs=-1,random_state=1019,init = 'k-means++')
#     kmodel.fit(finalDf)
#     inertia.append(np.sqrt(kmodel.inertia_))
# plt.plot(range(1,10),inertia,'o-')
# plt.xlabel('k')
# plt.show()
k = 2
kmodel = KMeans(n_clusters=k, random_state=1019, init='k-means++').fit(finalDf)
kprediction = kmodel.predict(finalDf)
print(kprediction)
kprediction = pd.DataFrame(kprediction)
kprediction.columns = ['GCs']
kmeans = pd.concat([principalDf, kprediction], axis=1)
print('-' * 50)
print(kmeans)
kmeans.to_csv('./kmeans.csv')
# np.savetxt("kmeans.csv", kmeans, delimiter=',', fmt='%d')

# 约束种子K-means
# class KMeans:
#     def __init__(self,k=2):
#         self.labels_=None
#         self.mu=None
#         self.k=k
#
#     def fit(self,X,L,U):
#         while True:
#             self.mu = np.zeros((self.k, L.shape[1]))
#             for j in range(self.k):
#                 self.mu[j] = np.mean(L[j], axis=0)
#             C = {}
#             D = {}
#             for j in range(self.k):
#                 C[j]=[]
#                 D[j]=[]
#             for j in range(self.k):
#                 for i in range(L.shape[0]):
#                     C[j].append(i)
#
#             for j in range(U.shape[0]):
#                 for i in range(self.k):
#                     d = np.sqrt(np.sum((U[j] - self.mu[i]) ** 2))
#                     r = np.argmin(d)
#                     D[r].append(j)
#
#             for j in range(self.k):
#                 self.mu[j] = np.mean(C[j],axis = 0)
#
#             self.labels_ = np.zeros((X.shape[0],), dtype=np.int32)
#             for i in range(self.k):
#                 self.labels_[C[i]] = i
#
#     def predict(self,X):
#         preds=[]
#         for j in range(X.shape[0]):
#             d=np.zeros((self.k,))
#             for i in range(self.k):
#                 d[i]=np.sqrt(np.sum((X[j]-self.mu[i])**2))
#             preds.append(np.argmin(d))
#         return np.array(preds)

# Autoencoder
seed = 7
np.random.seed(seed)
n_clusters = 10
BathSize = 1024
InCol = 4
OuCol = 1
TestSize = 0.33
Epochs = 5000

train = pd.read_csv('./kmeans.csv')
feature = ['0', '1', '2', '3']
x = train.loc[:, feature].values
y = train.loc[:, ['GCs']].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=TestSize, random_state=seed)
input_dim = Input(shape=(InCol,))
encoded = Dense(100, activation='relu')(input_dim)
encoded = Dense(50, activation='relu')(encoded)
encoded = Dense(20, activation='relu')(encoded)
encoded = Dense(n_clusters, activation=sigmoid)(encoded)

decoded = Dense(20, activation=relu)(encoded)
decoded = Dense(50, activation=relu)(decoded)
decoded = Dense(100, activation=relu)(decoded)
decoded = Dense(OuCol)(decoded)

autoencoder = Model(input_dim, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
train_history = autoencoder.fit(X_train, y_train, epochs=Epochs, batch_size=BathSize)
predicts = autoencoder.predict(X_test)
print(np.absolute(np.rint(predicts)))
