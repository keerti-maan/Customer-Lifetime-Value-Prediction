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


