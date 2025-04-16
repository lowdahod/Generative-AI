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




from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Enhanced legal and medical corpus
enhanced_corpus = [
    # Legal domain
    "The court ordered the immediate release of the detained individual due to lack of evidence.",
    "A new amendment was introduced to ensure the protection of intellectual property rights.",
    "The defendant pleaded not guilty, citing an alibi supported by credible witnesses.",
    "The plaintiff accused the company of violating environmental regulations.",
    "A settlement agreement was reached through arbitration, avoiding a lengthy trial.",
    "The legal team presented a compelling argument to overturn the previous judgment.",
    "Contractual obligations must be fulfilled unless waived by mutual consent.",
    "The jury found the accused guilty of fraud and embezzlement.",
    "The appeal was dismissed as the evidence presented was deemed inadmissible.",
    "The attorney emphasized the importance of adhering to constitutional rights.",

    # Medical domain
    "The patient was admitted to the emergency department with severe chest pain.",
    "The surgeon successfully performed a minimally invasive procedure to remove the tumor.",
    "Clinical trials showed significant improvement in patients treated with the experimental drug.",
    "Regular screening is essential for early detection of chronic illnesses such as diabetes.",
    "The doctor recommended physical therapy to improve mobility after surgery.",
    "The hospital implemented stringent protocols to prevent the spread of infectious diseases.",
    "The nurse monitored the patient's vital signs hourly to ensure stability.",
    "Vaccination campaigns have drastically reduced the prevalence of polio worldwide.",
    "The radiologist identified a small abnormality in the CT scan requiring further investigation.",
    "Proper nutrition and exercise are vital components of a healthy lifestyle."
]


tokenized_corpus = [simple_preprocess(sentence) for sentence in 
enhanced_corpus] 
tokenized_corpus 

tokenized_corpus = [simple_preprocess(sentence) for sentence in enhanced_corpus]

domain_word2vec = Word2Vec(
    sentences=tokenized_corpus,  # Tokenized text data
    vector_size=100,            # Dimensionality of the word embeddings
    window=5,                   # Context window size
    min_count=1,                # Include all words, even rare ones
    sg=1,                       # Use Skip-gram model for training
    epochs=150                  # Number of training iterations
)

# Save the trained model
domain_word2vec.save("enhanced_domain_word2vec.model")


# Analyze embeddings: Get vectors for specific words 
words_to_analyze = ["court", "plaintiff", "doctor", "patient", 
"guilty", "surgery"] 
for word in words_to_analyze: 
    if word in domain_word2vec.wv: 
        print(f"Vector embedding for '{word}':\n{domain_word2vec.wv[word]}\n") 
    else: 
        print(f"Word '{word}' not found in the Word2Vec model.")


selected_words = ["court", "plaintiff", "defendant", "guilty", "jury", 
"patient", "doctor", "hospital", "surgery", 
"emergency"] 
word_vectors = [domain_word2vec.wv[word] for word in selected_words] 
word_vectors 


pca = PCA(n_components=2) 
reduced_vectors = pca.fit_transform(word_vectors) 
reduced_vectors

plt.figure(figsize=(12, 8)) 
for i, word in enumerate(selected_words): 
    plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1]) 
    plt.text(reduced_vectors[i, 0] + 0.002, reduced_vectors[i, 1], 
    word, fontsize=12) 
plt.title("PCA Visualization of Legal and Medical Word Embeddings") 
plt.xlabel("PCA Dimension 1") 
plt.ylabel("PCA Dimension 2") 
plt.show()
