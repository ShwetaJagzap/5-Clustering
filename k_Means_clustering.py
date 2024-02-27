# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:42:36 2023

@author: ACER
"""

#K-means clustering
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
#let us try to understand first how k means work for two
#dimentional data
#for that generate random numbers in the range 0 to 1
#and with uniform distribution of 1/50
X=np.random.uniform (0,1,50)
Y=np.random.uniform(0,1,50)
#create a empty dataframe with 0 rows and 2 column 
df_xy=pd.DataFrame(columns=['X','Y'])
#assign  the values of x and y to these columns 
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x='X',y='Y',kind='scatter')
model1=KMeans(n_clusters=3).fit(df_xy)
'''with data x and y apply kmeans model, generate scatter plot:
    with scale/font=10'''
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)
Univ1=pd.read_csv('University_Clustering.csv')
Univ1.describe()
Univ=Univ1.drop(["State"],axis=1)
#we know that there is scale diffrence among the columns , which we have 
#either by using normalization or standardization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#now apply this normalization function tio univ datframe for all the row
df_norm=norm_func(Univ1.iloc[:,1:])
'''
what will be total cluster number, will be 1,2 or 3
'''

TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=1)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)#total within sum of square
    
    
TWSS
#as k value increases the TWSS value decreases
plt.plot(k,TWSS,'ro-')
plt.xlabel('No_of_clusters') ;
plt.ylabel("Total_within_SS")
'''
how to select value of k from elbow curve
when k changes from 2 to 3, decreases in twss is higher than 
when k changes from 3 to 4
when k values changes from 5 to 6 decreases 
in twss is considerably less hence considerde k=3
'''  
model=KMeans(n_cluster=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)  
Univ['clust']=mb
Univ.head()
Univ=Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).means()
Univ.to_csv('kmeans_Univesity.csv',encoding='utf-8')
import os 
