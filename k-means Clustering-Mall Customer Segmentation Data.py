#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import os
for dirname, _, filenames in os.walk(r'C:\Users\91944\OneDrive\Desktop\TASK\Mall_Customers.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # dataset overview

# Mall Customer Segmentation Data
# 
# Description: Contains age, income, and spending score of mall customers.
# 
# Source: Kaggle
# 
# Why?
# 
# Perfect for K-Means and Hierarchical Clustering.
# 
# Clear business use case (targeted marketing).
# 
# Clusters to Find: High-income spenders, frugal customers, young shoppers, etc.

# In[3]:


df = pd.read_csv(r'C:\Users\91944\OneDrive\Desktop\TASK\Mall_Customers.csv')
df.head()


# # Exploratory Data Analysis (EDA)

# Dataset Size

# In[4]:


rows,columns = df.shape
shp = pd.DataFrame({"": ["Rows", "Columns"], "Count": [rows, columns]})
shp.set_index("", inplace=True)
shp


# Missing Values in each column

# In[5]:


empty = pd.DataFrame(df.isna().sum())
empty.rename(columns={0:"Count of Empty Values"})


# Summary statistics

# In[6]:


df.describe()


# In[7]:


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['Annual Income (k$)'], kde=True, bins=20)
plt.title("Annual Income Distribution")

plt.subplot(1, 2, 2)
sns.histplot(df['Spending Score (1-100)'], kde=True, bins=20)
plt.title("Spending Score Distribution")
plt.show()


# Key Insights:
# 
# No missing values.
# 
# Income ranges from $15k to $137k, Spending Score from 1 to 100.

# # Feature Selection & Scaling

# K-Means and Hierarchical Clustering are distance-based algorithms; scaling ensures equal feature influence.

# In[8]:


# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data Sample:")
print(X_scaled[:5])


# # K-Means Clustering

# Find Optimal Clusters (Elbow Method)

# In[9]:


# Elbow method to find optimal k
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()


# Interpretation:
# 
# Look for the "elbow" (here, k=5).

# Train K-Means with k=5

# In[10]:


kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster_KMeans', palette='viridis', s=100)
plt.title("K-Means Clusters (k=5)")
plt.show()


# Cluster Descriptions:
# 
# Cluster 0: Low income, low spending (Budget-conscious).
# 
# Cluster 1: High income, low spending (Savvy spenders).
# 
# Cluster 2: Medium income, medium spending (Average).
# 
# Cluster 3: High income, high spending (Target customers).
# 
# Cluster 4: Low income, high spending (Carefree spenders).

# # Hierarchical Clustering

# Dendrogram to Find Optimal Clusters

# In[12]:


# Plot dendrogram
plt.figure(figsize=(10, 6))
dend = dendrogram(linkage(X_scaled, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()


# Interpretation:
# 
# Vertical line with the longest distance without horizontal lines suggests k=5.

# Train Agglomerative Clustering

# In[14]:


hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
df['Cluster_Hierarchical'] = hc.fit_predict(X_scaled)

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster_Hierarchical', palette='tab10', s=100)
plt.title("Hierarchical Clusters (k=5)")
plt.show()


# # Algorithm Comparison 

# In[15]:


# Silhouette Scores
kmeans_score = silhouette_score(X_scaled, df['Cluster_KMeans'])
hc_score = silhouette_score(X_scaled, df['Cluster_Hierarchical'])

comparison = pd.DataFrame({
    'Algorithm': ['K-Means', 'Hierarchical'],
    'Silhouette Score': [kmeans_score, hc_score]
})

print("\nAlgorithm Performance:")
display(comparison)


# Conclusion:
# 
# Both methods perform similarly (k=5 is optimal).
# 
# K-Means slightly edges out in compactness.

# # Cluster Interpretation (Business Insights)

# Cluster Profiles
# 
# High Income, Low Spending (Savvy): Target with loyalty programs.
# 
# High Income, High Spending (Target): Upsell premium products.
# 
# Low Income, High Spending (Carefree): Offer discounts to retain.
# 
# Low Income, Low Spending (Budget): Low-priority for marketing.
# 
# Medium Income, Medium Spending (Average): General promotions.

# Recommendations
# 
# Personalized Marketing: Tailor campaigns based on cluster traits.
# 
# Dynamic Pricing: Adjust for high-spending clusters.

# In[ ]:




