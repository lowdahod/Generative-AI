!pip install gensim scikit-learn matplotlib numpy


!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip

# Convert GloVe format to Word2Vec format
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = "glove.6B.100d.txt"
word2vec_output_file = "glove.6B.100d.word2vec.txt"
glove2word2vec(glove_input_file, word2vec_output_file)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Select words from a specific domain (sports in this example)
words = ['football', 'soccer', 'basketball', 'tennis', 'baseball', 
         'coach', 'goal', 'player', 'referee', 'team',
         'engineer', 'information']  # Added some non-sports words for contrast

# Get word vectors
word_vectors = np.array([model[word] for word in words if word in model])

# Filter words that exist in the model
valid_words = [word for word in words if word in model]

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(word_vectors)

# Plotting
plt.figure(figsize=(12, 10))
plt.scatter(pca_result[:, 0], pca_result[:, 1])

# Add labels
for i, word in enumerate(valid_words):
    plt.annotate(word, xy=(pca_result[i, 0], pca_result[i, 1]), 
                 xytext=(5, 2), textcoords='offset points',
                 ha='right', va='bottom')

plt.title("PCA Visualization of Word Embeddings")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.show()




# t-SNE often gives better separation of clusters
tsne = TSNE(n_components=2, random_state=42, perplexity=3)
tsne_result = tsne.fit_transform(word_vectors)

# Plotting
plt.figure(figsize=(12, 10))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])

# Add labels
for i, word in enumerate(valid_words):
    plt.annotate(word, xy=(tsne_result[i, 0], tsne_result[i, 1]), 
                 xytext=(5, 2), textcoords='offset points',
                 ha='right', va='bottom')

plt.title("t-SNE Visualization of Word Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()



def get_similar_words(word, model, topn=5):
    try:
        similar_words = model.most_similar(word, topn=topn)
        print(f"Words similar to '{word}':")
        for word, similarity in similar_words:
            print(f"{word}: {similarity:.4f}")
        return similar_words
    except KeyError:
        print(f"Word '{word}' not in vocabulary")
        return []

# Example usage
similar_words_football = get_similar_words('football', model)


