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