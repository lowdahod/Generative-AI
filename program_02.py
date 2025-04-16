import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

# Load the pre-trained GloVe embeddings (100D model)
model_100d = KeyedVectors.load_word2vec_format("/content/glove.6B.100d.word2vec.txt", binary=False, limit=500000)

# Select words from a specific domain (sports) and others
words = ['football', 'soccer', 'basketball', 'tennis', 'engineer', 
         'information', 'baseball', 'coach', 'goal', 'player', 
         'referee', 'team']

# Extract vectors for selected words
word_vectors = np.array([model_100d[word] for word in words])

# Perform PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(word_vectors)

# Plotting the words in 2D
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    plt.scatter(pca_result[i, 0], pca_result[i, 1])
    plt.text(pca_result[i, 0] + 0.02, pca_result[i, 1], word, fontsize=12)
plt.title("PCA Visualization of Word Embeddings")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.show()


# Function to find similar words
def get_similar_words(word, model, topn=5):
    similar_words = model.similar_by_word(word, topn=topn)
    return similar_words

# Example: Find words similar to 'football'
similar_words_football = get_similar_words('football', model_100d, topn=5)
print(f"Words similar to 'football': {similar_words_football}")


# List of words to print embeddings for
words_to_print = ['football', 'soccer']

# Print their embeddings
for word in words_to_print:
    if word in model_100d:
        print(f"Vector embedding for '{word}':\n{model_100d[word]}\n")
    else:
        print(f"Word '{word}' not found in the model.")
