# Install gensim if not already installed
!pip install gensim

# Import necessary libraries
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np

# Step 1: Convert GloVe to Word2Vec format and Load Models
from gensim.scripts.glove2word2vec import glove2word2vec 
from gensim.models import KeyedVectors 
 
# Paths to the GloVe file and output Word2Vec file 
glove_input_file = "/content/glove.6B.100d.txt"  # Path to GloVe file 
word2vec_output_file = "/content/glove.6B.100d.word2vec.txt"  # Output file in Word2Vec format 
 
# Convert GloVe format to Word2Vec format 
glove2word2vec(glove_input_file, word2vec_output_file) 
 
# Load the converted Word2Vec model 
model = KeyedVectors.load_word2vec_format(word2vec_output_file, 
binary=False) 
 
# Test the loaded model 
print(model.most_similar("king")) 

similar_to_mysore = model.similar_by_vector(model['mysore'], topn=5) 
print(f"Words similar to 'mysore': {similar_to_mysore}")

# Perform vector arithmetic 
result_vector_1 = model['actor'] - model['man'] + model['woman'] 
 
# Find the most similar word 
result_1 = model.similar_by_vector(result_vector_1, topn=1) 
print(f"'actor - man + woman' = {result_1}") 


# Perform vector arithmetic 
result_vector_2 = model['india'] - model['delhi'] + model['washington'] 
 
# Find the most similar word 
result_2 = model.similar_by_vector(result_vector_2, topn=3) 
print(f"'India - Delhi + Washington' = {result_2}")

scaled_vector = model['hotel'] * 2  # Scales the 'king' vector by a 
factor of 2 
result_2 = model.similar_by_vector(scaled_vector, topn=3) 
result_2 

import numpy as np 
normalized_vector = model['fish'] / np.linalg.norm(model['fish']) 
result_2 = model.similar_by_vector(normalized_vector, topn=3) 
result_2 

average_vector = (model['king'] + model['woman'] + model['man']) / 3 
result_2 = model.similar_by_vector(average_vector, topn=3) 
result_2 

glove_input_file = "/content/glove.6B.50d.txt"  # Path to GloVe file 
word2vec_output_file = "/content/glove.6B.50d.word2vec.txt"  # Output 
file in Word2Vec format 
 
# Convert GloVe format to Word2Vec format 
glove2word2vec(glove_input_file, word2vec_output_file) 
 
# Load the converted Word2Vec model 
model_50d = KeyedVectors.load_word2vec_format(word2vec_output_file, 
binary=False) 
 
# Paths to the GloVe file and output Word2Vec file 
glove_input_file = "/content/glove.6B.100d.txt"  # Path to GloVe file 
word2vec_output_file = "/content/glove.6B.100d.word2vec.txt"  # Output file in Word2Vec format 
 
# Convert GloVe format to Word2Vec format 
glove2word2vec(glove_input_file, word2vec_output_file) 
 
# Load the converted Word2Vec model 
model_100d = KeyedVectors.load_word2vec_format(word2vec_output_file, 
binary=False)

word1 = "hospital" 
word2 = "doctor" 
 
# Similarity in 50d 
similarity_50d = model_50d.similarity(word1, word2) 
 
# Similarity in 100d 
similarity_100d = model_100d.similarity(word1, word2) 
 
# Results 
print(f"Similarity (50d) between '{word1}' and '{word2}': 
{similarity_50d:.4f}") 
print(f"Similarity (100d) between '{word1}' and '{word2}': 
{similarity_100d:.4f}")

# Calculate distance between two words 
distance_50d = model_50d.distance(word1, word2) 
distance_100d = model_100d.distance(word1, word2) 
# Results 
print(f"Distance (50d) between '{word1}' and '{word2}': 
{distance_50d:.4f}") 
print(f"Distance (100d) between '{word1}' and '{word2}': 
{distance_100d:.4f}") 

