import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt
from IPython.display import Image

filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

print(len(filenames))
print(filenames[0])

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tf.keras.models.Sequential([model,
                                   GlobalMaxPool2D()
                                   ])
model.summary()

img = image.load_img('16871.jpg', target_size=(224,224))
img_array = image.img_to_array(img)
img_expand_dim = np.expand_dims(img_array, axis=0)
img_preprocess = preprocess_input(img_expand_dim)
result = model.predict(img_preprocess).flatten()
norm_result = result/norm(result)
norm_result

def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result/norm(result)
    return norm_result

extract_features_from_images(filenames[0], model)

image_features = []
for file in filenames:
    image_features.append(extract_features_from_images(file, model))
image_features

Image_features = pkl.dump(image_features, open('Images_features.pkl','wb'))
filenames = pkl.dump(filenames, open('filenames.pkl','wb'))
Image_features = pkl.load(open('Images_features.pkl','rb'))
filenames = pkl.load(open('filenames.pkl','rb'))
np.array(Image_features).shape

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)
input_image = extract_features_from_images('16871.jpg',model)
distance,indices = neighbors.kneighbors([input_image])
indices[0]
from IPython.display import Image
Image('16871.jpg')
Image(filenames[indices[0][1]])
Image(filenames[indices[0][2]])
Image(filenames[indices[0][3]])
Image(filenames[indices[0][4]])
Image(filenames[indices[0][5]])

import random


def retrieval_accuracy(filenames, model, n_samples=10, n_neighbors=6):
    correct = 0
    for _ in range(n_samples):
        # Randomly select an image
        query_index = random.randint(0, len(filenames) - 1)
        query_image = filenames[query_index]

        query_features = extract_features_from_images(query_image, model)

        distances, indices = neighbors.kneighbors([query_features])

        if query_index in indices[0]:
            correct += 1

    accuracy = correct / n_samples
    return accuracy


accuracy = retrieval_accuracy(filenames, model)
print(f"Retrieval Accuracy: {accuracy * 100:.2f}%")


def visualize_retrieval(filenames, model, neighbors, n_neighbors=6):
    query_index = random.randint(0, len(filenames) - 1)
    query_image = filenames[query_index]
    query_features = extract_features_from_images(query_image, model)
    distances, indices = neighbors.kneighbors([query_features])

    plt.figure(figsize=(15, 5))

    # query image
    plt.subplot(1, n_neighbors + 1, 1)
    plt.imshow(image.load_img(query_image))
    plt.title("Query Image")
    plt.axis('off')

    # Show nearest neighbors
    for i, idx in enumerate(indices[0]):
        plt.subplot(1, n_neighbors + 1, i + 2)
        plt.imshow(image.load_img(filenames[idx]))
        plt.title(f"Neighbor {i + 1}")
        plt.axis('off')

    plt.show()

# VisualizIing a random retrieval from the dataset
visualize_retrieval(filenames, model, neighbors)


import matplotlib.pyplot as plt

# Load the precomputed features
Image_features = pkl.load(open('Images_features.pkl','rb'))

# Convert to a NumPy array for easier manipulation
image_features_array = np.array(Image_features)

# Plotting histograms of the first few features
plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.hist(image_features_array[:, i], bins=20, color='skyblue')
    plt.title(f'Feature {i+1}')
plt.tight_layout()
plt.show()



image_features = pkl.load(open('Images_features.pkl', 'rb'))

# Reducing the dimensions to 2D for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(image_features)

# PLotting our 2D representation of the feature vectors
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', marker='o')

plt.title('PCA of Image Features')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# # If you want to annotate the points with filenames we can use this code in extra but as we have large set then this would make it clumpsy
# for i in range(len(filenames)):
#     plt.annotate(os.path.basename(filenames[i]), (pca_result[i, 0], pca_result[i, 1]))

plt.show()


import numpy as np

# Compute the magnitude of each feature vector
magnitude = np.linalg.norm(image_features, axis=1)

# Plot histogram
plt.figure(figsize=(10, 7))
plt.hist(magnitude, bins=30, color='blue', edgecolor='black')

plt.title('Histogram of Feature Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


image_features = np.array(image_features)

# correlation matrix
correlation_matrix = np.corrcoef(image_features, rowvar=False)


plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt='.2f')
plt.title('Heatmap of Feature Correlations')
plt.show()

#KNN algorithm with kd tree
from sklearn.neighbors import NearestNeighbors
import time
import matplotlib.pyplot as plt
from PIL import Image

# KNN with KD-Tree
start_time = time.time()

# Apply PCA to the entire Image_features (this reduces the dimensionality of your dataset)
Image_features_pca = pca.transform(Image_features)

# Train KNN on the PCA-transformed features
knn_kdtree = NearestNeighbors(n_neighbors=6, algorithm='kd_tree', metric='euclidean')
knn_kdtree.fit(Image_features_pca)  # Now fitting on the PCA-transformed features

# Extract features from the input image and transform them using the same PCA
input_image_feature = extract_features_from_images('16871.jpg', model)
input_image_pca = pca.transform([input_image_feature])  # Transforming input image feature using PCA

# Find the nearest neighbors for the PCA-transformed input image
distance, indices = knn_kdtree.kneighbors(input_image_pca)

# Plot the PCA results
plt.figure(figsize=(10, 7))

# Scatter plot for all the PCA-transformed image features
plt.scatter(Image_features_pca[:, 0], Image_features_pca[:, 1], c='blue', marker='o', label='Images')

# Highlight the nearest neighbors in red
plt.scatter(Image_features_pca[indices[0], 0], Image_features_pca[indices[0], 1], c='red', marker='x', s=100, label='Nearest Neighbors')

# Label the plot
plt.title('KNN (KD-Tree) with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

# Show the plot
plt.show()

# Display the recommended images
for i in indices[0]:
    display(Image.open(filenames[i]))  # Make sure 'filenames' list has the correct file paths

# Print the execution time for the KNN operation
print(f"Execution Time for KNN with KD-Tree: {time.time() - start_time} seconds")


# KNN with Ball Tree
start_time = time.time()
knn_balltree = NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric='euclidean')
knn_balltree.fit(pca_result)

# Find the nearest neighbors for the input image
distance, indices = knn_balltree.kneighbors(input_image_pca)

# Plot the PCA results
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', marker='o')
plt.scatter(pca_result[indices[0], 0], pca_result[indices[0], 1], c='red', marker='x', s=100)

plt.title('KNN (Ball Tree) with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Display recommended images
for i in indices[0]:
    display(Image.open(filenames[i]))

print(f"Execution Time for KNN with Ball Tree: {time.time() - start_time} seconds")




# Perform KMeans clustering on image features
n_clusters = 10  # You can adjust this number based on your dataset
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Start timing the clustering process
start_time = time.time()

# Fitting KMeans to the image features
kmeans.fit(Image_features)

# Predict the cluster for each image feature
cluster_assignments = kmeans.labels_

print(f"Execution Time for KMeans: {time.time() - start_time} seconds")

# Perform PCA for 2D visualization 
pca = PCA(n_components=2)
pca_result = pca.fit_transform(Image_features)

# Visualize the clusters with PCA
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_assignments, cmap='rainbow')
plt.title('KMeans Clustering of Image Features with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()


# Now we get recommendations for an input image based on its cluster
def recommend_similar_images(input_image_path, model, kmeans, filenames, n_recommendations=5):
    # Extract features from the input image
    input_image_feature = extract_features_from_images(input_image_path, model)

    # Predict the cluster of the input image
    input_image_cluster = kmeans.predict([input_image_feature])[0]

    # Get indices of images in the same cluster
    similar_images_indices = np.where(cluster_assignments == input_image_cluster)[0]

    # Recommend 'n_recommendations' images from the same cluster
    print(f"Images recommended from Cluster {input_image_cluster}:")
    for i in similar_images_indices[:n_recommendations]:
        display(Image(filenames[i]))


# Example: Recommending images similar to '16871.jpg'
recommend_similar_images('16871.jpg', model, kmeans, filenames, n_recommendations=5)
