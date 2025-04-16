!pip install gensim scikit-learn matplotlib numpy


from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Define your domain-specific corpus (legal domain in this example)
legal_corpus = [
    "The court ruled in favor of the plaintiff.",
    "The defendant was found guilty of negligence.",
    "A breach of contract case was filed.",
    "The agreement between parties must be honored.",
    "The lawyer presented compelling evidence.",
    "Legal documents must be drafted carefully.",
    "The jury deliberated for several hours.",
    "A settlement was reached between the parties.",
    "The plaintiff claimed damages for losses incurred.",
    "The contract outlined the obligations of both parties."
]

# Preprocess the corpus (tokenization)
tokenized_corpus = [simple_preprocess(sentence) for sentence in legal_corpus]
print(tokenized_corpus)

# Train the Word2Vec model
legal_word2vec = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=50,       # Embedding dimension
    window=3,            # Context window size
    min_count=1,         # Minimum word frequency
    sg=1,                # Skip-gram model (1 for skip-gram, 0 for CBOW)
    epochs=100           # Number of training iterations
)

# Save the model (optional)
legal_word2vec.save("legal_word2vec.model")



word = "court"
if word in legal_word2vec.wv:
    print(f"Vector embedding for '{word}':\n{legal_word2vec.wv[word]}\n")
else:
    print(f"Word '{word}' not found in the model.")


similar_words = legal_word2vec.wv.most_similar("court", topn=5)
print(f"Words similar to 'court': {similar_words}")



# Select words to visualize
words_to_visualize = ["court", "plaintiff", "defendant", "agreement", 
                     "lawyer", "evidence", "contract", "settlement", 
                     "jury", "damages"]

# Get vectors for selected words
word_vectors = [legal_word2vec.wv[word] for word in words_to_visualize]

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)

# Plot the embeddings
plt.figure(figsize=(12, 8))
for i, word in enumerate(words_to_visualize):
    plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
    plt.text(reduced_vectors[i, 0] + 0.02, reduced_vectors[i, 1], word, fontsize=12)
    
plt.title("PCA Visualization of Legal Word Embeddings")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.show()




enhanced_corpus = [
    "The court ruled in favor of the plaintiff.",
    "The patient underwent surgery for a critical condition.",
    "Legal documents must be carefully drafted to avoid disputes.",
    "The doctor prescribed medication for chronic illness.",
]

# Preprocess and train
tokenized_corpus = [simple_preprocess(sentence) for sentence in enhanced_corpus]
domain_word2vec = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,  # Higher dimensionality for richer representation
    window=5,
    sg=1,
    epochs=150
)

domain_word2vec.save("domain_word2vec.model")




