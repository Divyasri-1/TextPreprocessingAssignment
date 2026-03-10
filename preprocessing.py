
import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download resources if not already
nltk.download('punkt')
nltk.download('stopwords')

# Sample input text (two paragraphs)
text = """Artificial Intelligence (AI) is transforming industries rapidly! 
From healthcare, finance, and education; to entertainment, AI applications are everywhere. 
However, challenges such as bias, transparency, and accountability remain?

Data preprocessing is crucial: without it, models may fail. 
Cleaning, normalizing, and transforming data—these steps ensure accuracy. 
But, can preprocessing alone guarantee success? Not always; feature engineering plays a vital role too!
"""

print("Original Text:\n", text)

# Step 1: Tokenization
tokens = nltk.word_tokenize(text)

# Step 2: Stop word removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print("\nAfter Stop Word Removal:\n", " ".join(filtered_tokens))

# Step 3: Removing punctuation
no_punct = [word for word in filtered_tokens if word not in string.punctuation]
print("\nAfter Punctuation Removal:\n", " ".join(no_punct))

# Step 4: Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in no_punct]
print("\nAfter Stemming:\n", " ".join(stemmed))

# Step 5: Lemmatization
nlp = spacy.load("en_core_web_sm")
doc = nlp(" ".join(no_punct))
lemmatized = [token.lemma_ for token in doc]
print("\nAfter Lemmatization:\n", " ".join(lemmatized))
