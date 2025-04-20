# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:07:36 2025

@author: LAB
"""

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#Load model 
with open('kmeans_model.pkl','rb') as f:
    loaded_model = pickle.load(f)
    
#Set title 
st.title("K-Means Clustering Visualizer by Seatsiri Dangjanta")

#Ste the page config
st.set_page_config(page_title="K-Means Clustering", layout='centered')


#Load Dataset
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_cluster, cluster_std=0.60, random_state=0)

#Predict using the loaded model
y_kmeans = loaded_model.predict(X)

#plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red')
ax.title('k-Means Clustering')
ax.legend()
st.pyplot(fig)
#Display

