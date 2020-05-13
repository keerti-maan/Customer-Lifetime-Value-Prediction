# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


df=pd.read_excel('Online Retail.xlsx')
df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])

df_uk=df.query("Country=='United Kingdom'").reset_index(drop=True)

m3 = df_uk[(df_uk.InvoiceDate <pd.Timestamp(2011,6,1)) & (df_uk.InvoiceDate >= pd.Timestamp(2011,3,1))].reset_index(drop=True)
m6 = df_uk[(df_uk.InvoiceDate <pd.Timestamp(2011,12,1)) & (df_uk.InvoiceDate >pd.Timestamp(2011,6,1))].reset_index(drop=True)

df_user = pd.DataFrame(m3['CustomerID'].unique())
df_user.columns = ['CustomerID']

def order_cluster(cluster_field_name, target_field_name,DF,ascending):
    DF_new = DF.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    DF_new = DF_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    DF_new['index'] = DF_new.index
    DF_final = pd.merge(DF,DF_new[[cluster_field_name,'index']], on=cluster_field_name)
    DF_final = DF_final.drop([cluster_field_name],axis=1)
    DF_final = DF_final.rename(columns={"index":cluster_field_name})

    return DF_final

max_purchase = m3.groupby('CustomerID').InvoiceDate.max().reset_index()
max_purchase.columns = ['CustomerID','MaxPurchaseDate']
max_purchase['Recency'] = (max_purchase['MaxPurchaseDate'].max() - max_purchase['MaxPurchaseDate']).dt.days
df_user = pd.merge(df_user, max_purchase[['CustomerID','Recency']], on='CustomerID')

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4,init='k-means++')
kmeans.fit(df_user[['Recency']])
df_user['RecencyCluster'] = kmeans.predict(df_user[['Recency']])
df_user = order_cluster('RecencyCluster', 'Recency',df_user,False)

Frequency = m3.groupby('CustomerID').InvoiceDate.count().reset_index()
Frequency.columns = ['CustomerID','Frequency']
df_user = pd.merge(df_user, Frequency, on='CustomerID')

kmeans = KMeans(n_clusters=4,init='k-means++')
kmeans.fit(df_user[['Frequency']])
df_user['FrequencyCluster'] = kmeans.predict(df_user[['Frequency']])
df_user = order_cluster('FrequencyCluster', 'Frequency',df_user,False)

m3['Monetary'] = m3['UnitPrice'] * m3['Quantity']
Revenue = m3.groupby('CustomerID').Monetary.sum().reset_index()
df_user = pd.merge(df_user, Revenue, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(df_user[['Monetary']])
df_user['MonetaryCluster'] = kmeans.predict(df_user[['Monetary']])
df_user = order_cluster('MonetaryCluster', 'Monetary',df_user,True)

df_user['OverallScore'] = df_user['RecencyCluster'] + df_user['FrequencyCluster'] + df_user['MonetaryCluster']
df_user['Segment'] = 'Low-Value'
df_user.loc[df_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
df_user.loc[df_user['OverallScore']>4,'Segment'] = 'High-Value'

m6['Monetary'] = m6['UnitPrice'] * m6['Quantity']
df_user_6m = m6.groupby('CustomerID')['Monetary'].sum().reset_index()
df_user_6m.columns = ['CustomerID','m6_Monetary']

df_merge = pd.merge(df_user, df_user_6m, on='CustomerID', how='left')
df_merge =df_merge.fillna(0)


corr= df_merge.corr(method='pearson')

df_merge = df_merge[df_merge['m6_Monetary'] < df_merge['m6_Monetary'].quantile(0.99)]
kmeans = KMeans(n_clusters=3,init='k-means++')
kmeans.fit(df_merge[['m6_Monetary']])
df_merge['LTVCluster'] = kmeans.predict(df_merge[['m6_Monetary']])
df_merge = order_cluster('LTVCluster', 'm6_Monetary',df_merge,True)
df_cluster = df_merge.copy()
df_cluster.groupby('LTVCluster')['m6_Monetary'].describe()


from sklearn.model_selection import train_test_split

df_class = pd.get_dummies(df_cluster)

corr_matrix= df_class.corr(method='pearson')
X = df_class.drop(['LTVCluster','m6_Monetary'],axis=1)
y = df_class['LTVCluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#############################################################################################################
##### XGBoost
import xgboost as xgb
from sklearn.metrics import classification_report,confusion_matrix
from xgboost import XGBClassifier    
xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1,objective= 'multi:softprob',n_jobs=-1).fit(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'
         .format(xgb_model.score(X_train, y_train)))
print('*'*60)

print('Accuracy of XGB classifier on test set: {:.2f}'
        .format(xgb_model.score(X_test[X_train.columns], y_test)))
print('*'*60)

y_pred = xgb_model.predict(X_test)

print(classification_report(y_test, y_pred))

################################################################################
### ANN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


 
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

classifier = Sequential()
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'softmax'))
categorical_labels = to_categorical(y_train, num_classes=3)
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, categorical_labels, batch_size = 32, nb_epoch = 200)
y_pred = classifier.predict(X_test)
y_pred=np.argmax(y_pred, axis=1)

cm=confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))


