# Install gensim if not already installed
!pip install gensim

# Import necessary libraries
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np

# Step 1: Convert GloVe to Word2Vec format and Load Models
def convert_and_load(glove_path, word2vec_path):
    # Convert GloVe format to Word2Vec format
    glove2word2vec(glove_path, word2vec_path)
    # Load the converted Word2Vec model
    model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    return model

# File paths for GloVe embeddings
glove_50d_file = "/content/glove.6B.50d.txt"
word2vec_50d_file = "/content/glove.6B.50d.word2vec.txt"

glove_100d_file = "/content/glove.6B.100d.txt"
word2vec_100d_file = "/content/glove.6B.100d.word2vec.txt"

# Load models
model_50d = convert_and_load(glove_50d_file, word2vec_50d_file)
model_100d = convert_and_load(glove_100d_file, word2vec_100d_file)

# Step 2: Explore Word Relationships
# Example 1: Find Similar Words
print("Words similar to 'mysore' (100D model):")
print(model_100d.similar_by_vector(model_100d['mysore'], topn=5))

# Example 2: Gender Analogy (king - man + woman = queen)
result_vector = model_100d['actor'] - model_100d['man'] + model_100d['woman']
print("'actor - man + woman' =")
print(model_100d.similar_by_vector(result_vector, topn=1))

# Example 3: Country-City Relationship (India - Delhi + Washington)
result_vector = model_100d['india'] - model_100d['delhi'] + model_100d['washington']
print("'India - Delhi + Washington' =")
print(model_100d.similar_by_vector(result_vector, topn=3))

# Step 3: Perform Arithmetic Operations
# Scaling Vectors
scaled_vector = model_100d['hotel'] * 2
print("Scaled vector (hotel * 2):")
print(model_100d.similar_by_vector(scaled_vector, topn=3))

# Normalizing Vectors
normalized_vector = model_100d['fish'] / np.linalg.norm(model_100d['fish'])
print("Normalized vector (fish):")
print(model_100d.similar_by_vector(normalized_vector, topn=3))

# Averaging Vectors
average_vector = (model_100d['king'] + model_100d['woman'] + model_100d['man']) / 3
print("Average vector (king + woman + man) / 3:")
print(model_100d.similar_by_vector(average_vector, topn=3))

# Step 4: Compare Similarity and Distance
word1 = "hospital"
word2 = "doctor"

# Similarity Comparison
similarity_50d = model_50d.similarity(word1, word2)
similarity_100d = model_100d.similarity(word1, word2)
print(f"Similarity (50D) between '{word1}' and '{word2}': {similarity_50d:.4f}")
print(f"Similarity (100D) between '{word1}' and '{word2}': {similarity_100d:.4f}")

# Distance Comparison
distance_50d = model_50d.distance(word1, word2)
distance_100d = model_100d.distance(word1, word2)
print(f"Distance (50D) between '{word1}' and '{word2}': {distance_50d:.4f}")
print(f"Distance (100D) between '{word1}' and '{word2}': {distance_100d:.4f}")
