# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:25:03 2023

@author: ACER
"""

import pandas as pd
import numpy as np
uni1=pd.read_csv('University_Clustering.csv')
uni1.describe()
uni1.info()
uni=uni1.drop(["State"],axis=1)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#Considering only numerical data
uni_normal=scale(uni1.data)
uni_normal

pca=PCA(n_components=6)
pca_values=pca.fir_transform(uni_normal)

#The amount of variance that each PCA explains is
var=pca.explained_variance_ratio_
var

#PCA weights
#pca.components_
#pca.components_[0]


#Cimulative variance
var1=np.cumsum(np.round(var,decimals=4)*100)
var1

#varience plot for PCA componenets obtaind
plt.plot(var,color='red')
#PCA scores
pca_values
pca_data=pd.DataFrame(pca_values)
pca_data.columns='comp0','comp1','comp2','comp3','comp4','comp5'
final1=pd.concat([uni],[0:])
final1 = pd.concat([uni], axis=0)
