# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 00:01:23 2023

@author: urmii
"""
#3.	Perform clustering analysis on the telecom data set. 
#The data is a mixture of both categorical and numerical data. 
#It consists of the number of customers who churn out. 
#Derive insights and get possible information on factors that may affect the churn decision. 
#Refer to Telco_customer_churn.xlsx dataset.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#IMPORT DATA SET AND CREATE DATAFRAME
tele = pd.read_excel('C:/2-Datasets/Telco_customer_churn.xlsx')



tele.columns
tele.shape
#(3999, 12)

tele.describe()

#initially we will perform EDA to analyse the data

#pairplot
import seaborn as sns
plt.close();
sns.set_style("whitegrid");
sns.pairplot(tele, hue="Award?", height=3);
plt.show()

#pdf and cdf

counts, bin_edges = np.histogram(tele['Phone Service'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
'''
from pdf we can say that approx 90% of data have balance 20000
'''
plt.plot(bin_edges[1:], cdf)
plt.show();

#Boxplot and outliers treatment

sns.boxplot(tele['Count'])
sns.boxplot(tele['Quarter'])
sns.boxplot(tele['Online Security'])
sns.boxplot(tele['Online Backup'])
sns.boxplot(tele['Number of Referrals'])
sns.boxplot(tele['Monthly Charge'])
sns.boxplot(tele['Total Extra Data Charges'])
sns.boxplot(tele['Unlimited Data'])


'''
from box plot except cc2 miles, days since enroll and award? 
all other colmns have outliers
we need to remove them
'''
#1
iqr = tele['Quarter'].quantile(0.75)-tele['Quarter'].quantile(0.25)
iqr
q1=tele['Quarter'].quantile(0.25)
q3=tele['Quarter'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
tele['Quarter'] =  np.where(tele['Quarter']>u_limit,u_limit,np.where(tele['Quarter']<l_limit,l_limit,tele['Quarter']))
sns.boxplot(tele['Quarter'])

#2
iqr = tele['Number of Referrals'].quantile(0.75)-tele['Number of Referrals'].quantile(0.25)
iqr
q1=tele['Number of Referrals'].quantile(0.25)
q3=tele['Number of Referrals'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
tele['Number of Referrals'] =  np.where(tele['Number of Referrals']>u_limit,u_limit,np.where(tele['Number of Referrals']<l_limit,l_limit,tele['Number of Referrals']))
sns.boxplot(tele['Number of Referrals'])

#3
iqr = tele['Streaming Movies'].quantile(0.75)-tele['Streaming Movies'].quantile(0.25)
iqr
q1=tele['Streaming Movies'].quantile(0.25)
q3=tele['Streaming Movies'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
tele['Streaming Movies'] =  np.where(tele['Streaming Movies']>u_limit,u_limit,np.where(tele['Streaming Movies']<l_limit,l_limit,tele['Streaming Movies']))
sns.boxplot(tele['Streaming Movies'])

#now describe dataset
tele.describe()
#we can see that there is huge difference between min,max and mean
# values for all the columns so we need to normalize the dataset


t_data = pd.read_excel('C:/2-Datasets/Telco_customer_churn.xlsx')
t_data.describe()
t_data.columns
t_data.dtypes
t_data.shape

t_data.drop(['Customer ID','Count','Quarter'],axis=1,inplace=True)
#get dummy variables from data set
df_n = pd.get_dummies(t_data)

df_n.shape
df_n

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return x
df_norm =norm_func(df_n.iloc[:,1:])
b=df_norm.describe()

#now all data is normalized
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method='complete',metric='euclidean')
plt.figure(figsize=(25,20));
plt.title("Hierarchical clustering dendrogram")
plt.xlabel('Index')
plt.ylabel('Distance')

sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#now apply clustering 
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_norm)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to autos dataframe as column
t_data['cluster'] = cluster_labels
t_data.columns
#t_dataNew = t_data.iloc[-1]+ t_data.iloc[:,[i for i in range(0,28)]]
#t_dataNew.columns
t_data.iloc[:,2:].groupby(t_data.cluster).mean()
t_data.to_csv("Telco_customer_churnNew.csv",encoding='utf-8')
t_data.cluster.value_counts()
import os
os.getcwd()
