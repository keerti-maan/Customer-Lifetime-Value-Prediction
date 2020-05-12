# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel('Online Retail.xlsx')
dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'])
dataset['InvoiceYearMonth']=dataset['InvoiceDate'].map(lambda date : 100*date.year+date.month)
dataset['MonetaryValue']=dataset['Quantity']*dataset['UnitPrice']
monetary=dataset.groupby(['InvoiceYearMonth'])['MonetaryValue'].sum().reset_index()

uk=dataset.query("Country=='United Kingdom'").reset_index(drop=True)
users=pd.DataFrame((dataset['CustomerID']).unique())
users.columns=['CustomerID']

max_purch=uk.groupby('CustomerID').InvoiceDate.max().reset_index()
max_purch.columns = ['CustomerID','MaxPurchaseDate']
max_purch['Recency'] = (max_purch['MaxPurchaseDate'].max() - max_purch['MaxPurchaseDate']).dt.days
users = pd.merge(users, max_purch[['CustomerID','Recency']], on='CustomerID')

from sklearn.cluster import KMeans
wcss=[]
recency=users[['Recency']]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(recency)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans=KMeans(n_clusters=4 , init='k-means++')
kmeans.fit(users[['Recency']])
users['RecencyCluster']=kmeans.predict(users[['Recency']])

def order_cluster(cluster_name,target_name,data,ascending):
    data_new = data.groupby(cluster_name)[target_name].mean().reset_index()
    data_new = data_new.sort_values(by=target_name,ascending=ascending).reset_index(drop=True)
    data_new['index'] = data_new.index
    data_final = pd.merge(data,data_new[[cluster_name,'index']], on=cluster_name)
    data_final = data_final.drop([cluster_name],axis=1)
    data_final = data_final.rename(columns={"index":cluster_name})
    return data_final
users = order_cluster('RecencyCluster', 'Recency',users,True)

users.groupby('RecencyCluster')['Recency'].describe()

###################################################################################
# Calculating Frequency

frequency=uk.groupby(by='CustomerID').InvoiceDate.count().reset_index()
frequency.columns=['CustomerID','Frequency']

users=pd.merge(users,frequency,on='CustomerID')

kmeans=KMeans(n_clusters=4 , init='k-means++')
kmeans.fit(users[['Frequency']])
users['FrequencyCluster']=kmeans.predict(users[['Frequency']])

users = order_cluster('FrequencyCluster', 'Frequency',users,True)

users.groupby('FrequencyCluster')['Frequency'].describe()


###################################################################################

#Calculating Monetary


monetary_value = uk.groupby('CustomerID').MonetaryValue.sum().reset_index()
monetary_value.columns=['CustomerID','Monetary']
users = pd.merge(users, monetary_value, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(users[['Monetary']])
users['MonetaryCluster'] = kmeans.predict(users[['Monetary']])

users = order_cluster('MonetaryCluster', 'Monetary',users,True)

users.groupby('MonetaryCluster')['Monetary'].describe()

users['OverallScore'] = users['RecencyCluster'] + users['FrequencyCluster'] + users['MonetaryCluster']
users.groupby('OverallScore')['Recency','Frequency','Monetary'].mean()

users['Segment'] = 'Low-Value'
users.loc[users['OverallScore'] > 2,'Segment'] = 'Mid-Value' 
users.loc[users['OverallScore'] > 4,'Segment'] = 'High-Value'