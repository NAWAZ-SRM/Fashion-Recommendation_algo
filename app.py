import numpy as np
import pickle as pkl
import tensorflow as tf
tf.compat.v1.reset_default_graph()
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy.linalg import norm
import matplotlib

matplotlib.use('Agg')  # For non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import random
import time
import os

# Load precomputed features and filenames
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))


# Extract features from image function
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result


# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Apply PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
Image_features_pca = pca.fit_transform(Image_features)

# KNN models
knn_brute = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
knn_brute.fit(Image_features)

knn_kdtree = NearestNeighbors(n_neighbors=6, algorithm='kd_tree', metric='euclidean')
knn_kdtree.fit(Image_features_pca)

knn_balltree = NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric='euclidean')
knn_balltree.fit(Image_features_pca)

# KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(Image_features)
cluster_assignments = kmeans.labels_

# Streamlit UI
st.header("Fashion Recommendation System")

# Dropdown for selecting the algorithm
algorithm = st.selectbox("Select an Algorithm",
                         ['KNN', 'KNN with KDTree', 'KNN with Ball Tree', 'KMeans'])

# Image file uploader
upload_file = st.file_uploader("Upload an Image")

if upload_file is not None:
    # Save uploaded file temporarily
    with open(os.path.join('upload', upload_file.name), 'wb') as f:
        f.write(upload_file.getbuffer())
    st.subheader('Uploaded Image')
    st.image(upload_file)

    # Extract features from uploaded image
    input_img_features = extract_features_from_images(upload_file, model)

    if algorithm == 'KNN':
        distance, indices = knn_brute.kneighbors([input_img_features])
        st.subheader("Recommended Images")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][1]])
        with col2:
            st.image(filenames[indices[0][2]])
        with col3:
            st.image(filenames[indices[0][3]])
        with col4:
            st.image(filenames[indices[0][4]])
        with col5:
            st.image(filenames[indices[0][5]])

        # Plot feature histograms
        image_features_array = np.array(Image_features)
        plt.figure(figsize=(15, 5))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            plt.hist(image_features_array[:, i], bins=20, color='skyblue')
            plt.title(f'Feature {i + 1}')
        plt.tight_layout()
        st.pyplot(plt)

        # Heatmap of feature correlations
        correlation_matrix = np.corrcoef(image_features_array, rowvar=False)
        plt.figure(figsize=(10, 7))
        sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
        plt.title('Heatmap of Feature Correlations')
        st.pyplot(plt)

        # Magnitude of feature vectors histogram
        magnitude = np.linalg.norm(image_features_array, axis=1)
        plt.figure(figsize=(10, 7))
        plt.hist(magnitude, bins=30, color='blue', edgecolor='black')
        plt.title('Histogram of Feature Magnitudes')
        st.pyplot(plt)

    elif algorithm == 'KNN with KDTree':
        input_image_pca = pca.transform([input_img_features])
        distance, indices = knn_kdtree.kneighbors(input_image_pca)

        st.subheader("Recommended Images")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][1]])
        with col2:
            st.image(filenames[indices[0][2]])
        with col3:
            st.image(filenames[indices[0][3]])
        with col4:
            st.image(filenames[indices[0][4]])
        with col5:
            st.image(filenames[indices[0][5]])

        # PCA plot
        plt.figure(figsize=(10, 7))
        plt.scatter(Image_features_pca[:, 0], Image_features_pca[:, 1], c='blue', marker='o')
        plt.scatter(Image_features_pca[indices[0], 0], Image_features_pca[indices[0], 1], c='red', marker='x', s=100)
        plt.title('KNN (KD-Tree) with PCA')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        st.pyplot(plt)

    elif algorithm == 'KNN with Ball Tree':
        input_image_pca = pca.transform([input_img_features])
        distance, indices = knn_balltree.kneighbors(input_image_pca)

        st.subheader("Recommended Images")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][1]])
        with col2:
            st.image(filenames[indices[0][2]])
        with col3:
            st.image(filenames[indices[0][3]])
        with col4:
            st.image(filenames[indices[0][4]])
        with col5:
            st.image(filenames[indices[0][5]])

        # PCA plot
        plt.figure(figsize=(10, 7))
        plt.scatter(Image_features_pca[:, 0], Image_features_pca[:, 1], c='blue', marker='o')
        plt.scatter(Image_features_pca[indices[0], 0], Image_features_pca[indices[0], 1], c='red', marker='x', s=100)
        plt.title('KNN (Ball Tree) with PCA')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        st.pyplot(plt)

    elif algorithm == 'KMeans':
        input_image_cluster = kmeans.predict([input_img_features])[0]
        similar_images_indices = np.where(cluster_assignments == input_image_cluster)[0]

        st.subheader(f"Recommended Images from Cluster {input_image_cluster}")
        col1, col2, col3, col4, col5 = st.columns(5)
        for i in range(5):
            with locals()[f'col{i + 1}']:
                st.image(filenames[similar_images_indices[i]])

        # KMeans PCA plot
        plt.figure(figsize=(10, 7))
        plt.scatter(Image_features_pca[:, 0], Image_features_pca[:, 1], c=cluster_assignments, cmap='rainbow')
        plt.title('KMeans Clustering of Image Features with PCA')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        st.pyplot(plt)

