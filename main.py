import matplotlib
import pandas as pd
from numpy import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
# 查看降维后的数据
principalDf = pd.DataFrame(data=principalComponents)
finalDf = pd.concat([principalDf, df[['GCs']]], axis=1)
print(finalDf.head(5))
finalDf.to_csv("./pca.csv")