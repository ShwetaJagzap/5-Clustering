# -*- coding: utf-8 -*-


#4.	Perform clustering on mixed data. Convert the categorical
# variables to numeric by using dummies or label encoding
# and perform normalization techniques. The data set consists 
#of details of customers related to their auto insurance. 
#Refer to Autoinsurance.csv dataset.

'''
This dataset is about 9134 customers which have taken 
vehicle insurance.
The aim of this analysis to get know whether our insurance 
customers will extend their vehicle insurance based on their 
behaviour.
'''

#business objective
'''
business objective is to perform clustering on customers
based on their similar characteristics

'''

#dataset AUTO INSURANCE.CSV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


autoIns = pd.read_csv('AutoInsurance.csv')

autoIns.columns
00+'''
Index(['Customer', 'State', 'Customer Lifetime Value', 'Response', 'Coverage',
       'Education', 'Effective To Date', 'EmploymentStatus', 'Gender',
       'Income', 'Location Code', 'Marital Status', 'Monthly Premium Auto',
       'Months Since Last Claim', 'Months Since Policy Inception',
       'Number of Open Complaints', 'Number of Policies', 'Policy Type',
       'Policy', 'Renew Offer Type', 'Sales Channel', 'Total Claim Amount',
       'Vehicle Class', 'Vehicle Size'],
      dtype='object')'''

autoIns.dtypes
'''
DATA DICTIONARY
autoIns.dtypes
Out[154]: 
Customer                          object
State                             object
Customer Lifetime Value          float64
Response                          object
Coverage                          object
Education                         object
Effective To Date                 object
EmploymentStatus                  object
Gender                            object
Income                             int64
Location Code                     object
Marital Status                    object
Monthly Premium Auto               int64
Months Since Last Claim            int64
Months Since Policy Inception      int64
Number of Open Complaints          int64
Number of Policies                 int64
Policy Type                       object
Policy                            object
Renew Offer Type                  object
Sales Channel                     object
Total Claim Amount               float64
Vehicle Class                     object
Vehicle Size                      object
dtype: object
'''
#------------------------------------------------------------------------
#most of colmns are of object type so we need to convert 
# them to numeric using dummies



autoIns
autoIns.describe()
'''
autoIns
Out[155]: 
     Customer       State  ...  Vehicle Class Vehicle Size
0     BU79786  Washington  ...   Two-Door Car      Medsize
1     QZ44356     Arizona  ...  Four-Door Car      Medsize
2     AI49188      Nevada  ...   Two-Door Car      Medsize
3     WW63253  California  ...            SUV      Medsize
4     HB64268  Washington  ...  Four-Door Car      Medsize
      ...         ...  ...            ...          ...
9129  LA72316  California  ...  Four-Door Car      Medsize
9130  PK87824  California  ...  Four-Door Car      Medsize
9131  TD14365  California  ...  Four-Door Car      Medsize
9132  UP19263  California  ...  Four-Door Car        Large
9133  Y167826  California  ...   Two-Door Car      Medsize

[9134 rows x 24 columns]'''

#-----------------------------------------------------------------
autoIns.shape
#(9134, 24)

# PDF and CDF
counts, bin_edges = np.histogram(autoIns['Customer Lifetime Value'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
#---------------------------------------------------------------------
#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
#from  plots we can say that 80% of data have cust lifetime val
#less that 10000
#cust having cust lifetime val between 18000 to 27000 are approx 15%
plt.show();
#--------------------------------------------------------------------
#Boxplot and outlier treatment

sns.boxplot(autoIns['Customer Lifetime Value'])
sns.boxplot(autoIns['Income'])
sns.boxplot(autoIns['Monthly Premium Auto'])
sns.boxplot(autoIns['Months Since Last Claim'])
sns.boxplot(autoIns['Months Since Policy Inception'])
sns.boxplot(autoIns['Number of Open Complaints'])
sns.boxplot(autoIns['Number of Policies'])
sns.boxplot(autoIns['Total Claim Amount'])

#out of these income , Months Since Last Claim, Months Since Policy Inception
#do not have outliers
#--------------------------------------------------------------------
#DATA PREPROCESSING
#we need to remove outliers from other cols
#1
iqr = autoIns['Customer Lifetime Value'].quantile(0.75)-autoIns['Customer Lifetime Value'].quantile(0.25)
iqr

q1 = autoIns['Customer Lifetime Value'].quantile(0.25)
q3 = autoIns['Customer Lifetime Value'].quantile(0.75)

l_limit = q1-(1.5*iqr)
u_limit = q3+(1.5*iqr)

autoIns['Customer Lifetime Value'] = np.where(autoIns['Customer Lifetime Value'] >u_limit,u_limit,np.where(autoIns['Customer Lifetime Value']<l_limit,l_limit,autoIns['Customer Lifetime Value']))
sns.boxplot(autoIns['Customer Lifetime Value'])
#--------------------------------------------------------------------
#2
iqr = autoIns['Monthly Premium Auto'].quantile(0.75)-autoIns['Monthly Premium Auto'].quantile(0.25)
iqr

q1 = autoIns['Monthly Premium Auto'].quantile(0.25)
q3 = autoIns['Monthly Premium Auto'].quantile(0.75)

l_limit = q1-(1.5*iqr)
u_limit = q3+(1.5*iqr)

autoIns['Monthly Premium Auto'] = np.where(autoIns['Monthly Premium Auto'] >u_limit,u_limit,np.where(autoIns['Monthly Premium Auto']<l_limit,l_limit,autoIns['Monthly Premium Auto']))
sns.boxplot(autoIns['Monthly Premium Auto'])
#--------------------------------------------------------------------
#3
iqr = autoIns['Number of Open Complaints'].quantile(0.75)-autoIns['Number of Open Complaints'].quantile(0.25)
iqr

q1 = autoIns['Number of Open Complaints'].quantile(0.25)
q3 = autoIns['Number of Open Complaints'].quantile(0.75)

l_limit = q1-(1.5*iqr)
u_limit = q3+(1.5*iqr)

autoIns['Number of Open Complaints'] = np.where(autoIns['Number of Open Complaints'] >u_limit,u_limit,np.where(autoIns['Number of Open Complaints']<l_limit,l_limit,autoIns['Number of Open Complaints']))
sns.boxplot(autoIns['Number of Open Complaints'])
#-------------------------------------------------------------------
#4
iqr = autoIns['Number of Policies'].quantile(0.75)-autoIns['Number of Policies'].quantile(0.25)
iqr

q1 = autoIns['Number of Policies'].quantile(0.25)
q3 = autoIns['Number of Policies'].quantile(0.75)

l_limit = q1-(1.5*iqr)
u_limit = q3+(1.5*iqr)

autoIns['Number of Policies'] = np.where(autoIns['Number of Policies'] >u_limit,u_limit,np.where(autoIns['Number of Policies']<l_limit,l_limit,autoIns['Number of Policies']))
sns.boxplot(autoIns['Number of Policies'])
#--------------------------------------------------------------------
#5

iqr = autoIns['Total Claim Amount'].quantile(0.75)-autoIns['Total Claim Amount'].quantile(0.25)
iqr

q1 = autoIns['Total Claim Amount'].quantile(0.25)
q3 = autoIns['Total Claim Amount'].quantile(0.75)

l_limit = q1-(1.5*iqr)
u_limit = q3+(1.5*iqr)

autoIns['Total Claim Amount'] = np.where(autoIns['Total Claim Amount'] >u_limit,u_limit,np.where(autoIns['Total Claim Amount']<l_limit,l_limit,autoIns['Total Claim Amount']))
sns.boxplot(autoIns['Total Claim Amount'])


autoIns.describe()
#-------------------------------------------------------------------
#initially we need to drop some columns which does not have much use
#like State, Education, Marital Status, sales channel

autoIns.drop(['State','Education','Customer','Marital Status','Sales Channel'],axis=1, inplace=True)
autoIns.describe()
autoIns.shape
#There are still some object types in dataset so we need to 
#-------------------------------------------------------------------
#get dummy variables from data set

df_n= pd.get_dummies(autoIns)
df_n.shape
#---------------------------------------------------------------------
#now we have dataset df_n with all dtypes int
df_n.describe() 
#there is huge difference between min, max,and mean values in dataset cols
#so we need to normalize this data

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

df_normal = norm_func(df_n)
desc = df_normal.describe()
desc
#MODEL BUILDING
#now all data is normalized
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df_normal,method='complete',metric='euclidean')
plt.figure(figsize=(15,8))
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('Distance')
#ref of dendrogram

sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#now apply clustering 
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,
                                     linkage='complete',
                                     affinity='euclidean').fit(df_normal)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to autoIns dataframe as column
autoIns['cluster'] = cluster_labels

autoInsNew = autoIns.iloc[:,[-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
autoInsNew.iloc[:,2:].groupby(autoInsNew.cluster).mean()

autoInsNew.to_csv("AutoInsuranceNew.csv",encoding='utf-8')
import os
os.getcwd()