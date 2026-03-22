# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 00:29:27 2026

@author: Manny
"""
# 1 Tokenization
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

text = """Text mining is the process of extracting useful information from large volumes of unstructured text data."""

# Lowercase
text = text.lower()

# Tokenization
tokens = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if w not in stop_words]

# Stemming
ps = PorterStemmer()
tokens = [ps.stem(w) for w in tokens]

print(tokens)

#2 •Remove numbers and special symbols
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text2 = "Data mining in 2024 is growing rapidly!"

text2 = re.sub(r'[^a-zA-Z\s]', '', text2)

text2 = text2.lower()
print(text2)

tokens2 = word_tokenize(text2)

stopwords2 = set(stopwords.words('english'))
tokens2 = [w for w in tokens2 if w not in stopwords2]

print(tokens2)

#3 Natural Language Content Analysis by calculating the frequency distribution of words 

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

text3 = "Social media platforms generate  large volumes of textual data."

tokens3 = word_tokenize(text3.lower())

fd3 = FreqDist(tokens3)

print(fd3.most_common(5))

#4 perform tokenization and POS tagging, then count the number of:NounsVerbsAdjectives

import nltk
nltk.download('averaged_perceptron_tagger_eng')
text4 = "The young athlete delivered a remarkable performance."

tokens4 = nltk.word_tokenize(text4)

tags4 = nltk.pos_tag(tokens4)

noun4 = verb4 = adj4 = 0

for word, tag in tags4:
    if tag.startswith('NN'):
        noun4 += 1
    elif tag.startswith('VB'):
        verb4 += 1
    elif tag.startswith('JJ'):
        adj4 += 1

print(noun4, verb4, adj4)

#5Python program to remove punctuation and stop words 

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text5 = "The weather forecast predicts heavy rainfall tomorrow."

tokens5 = word_tokenize(text5.lower())

tokens5 = [w for w in tokens5 if w not in string.punctuation]

stopwords5 = set(stopwords.words('english'))
tokens5 = [w for w in tokens5 if w not in stopwords5]

print(tokens5)

#6compute Term Frequency (TF) for each word in the following document.
from collections import Counter

text6 = "Local newspapers reported that the football team won an important match yesterday evening."

words6 = text6.split()

tf6 = Counter(words6)

print(tf6)


#7word frequency and document frequency 
docs7 = [
    "city hosted sports event",
    "sports event attracted players",
    "citizens enjoyed competition"
]

from collections import Counter

wf7 = Counter(" ".join(docs7).split())

df7 = {}
for doc in docs7:
    for word in set(doc.split()):
        df7[word] = df7.get(word, 0) + 1
        
print("wf",wf7)
print("df",df7)

#8program to perform sentiment analysis  lexicon-based approach.
positive8 = ["excellent", "great", "amazing"]
negative8 = ["poor", "bad"]

reviews8 = [
    "excellent camera great battery",
    "performance poor battery",
    "display amazing design"
]

for r in reviews8:
    score8 = 0
    for w in r.split():
        if w in positive8:
            score8 += 1
        elif w in negative8:
            score8 -= 1

    print("Positive" if score8 > 0 else "Negative")
    
    
#9 Python program to implement a sentiment analysis model using a Naive Bayes classifie
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

docs9 = [
    "fantastic entertaining",
    "disappointing poorly",
    "performed well enjoyed",
    "badly unhappy",
    "exciting match",          # NEW
    "great exciting game"      # NEW
]

labels9 = ["pos", "neg", "pos", "neg", "pos", "pos"]

# Vectorization
cv9 = CountVectorizer()
X9 = cv9.fit_transform(docs9)

# Model training
model9 = MultinomialNB()
model9.fit(X9, labels9)

# Testing
test9 = cv9.transform(["exciting match"])
print(model9.predict(test9))



#10 Python program to identify frequently occurring topics from the following news 
from collections import Counter

news10 = [
    "government economic policies",
    "football team championship",
    "economic growth increased",
    "championship match fans"
]

words10 = " ".join(news10).split()

freq10 = Counter(words10)

print(freq10.most_common(5))






#11 Text Classification (TF-IDF + Naive Bayes)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

docs11 = [
    "football team won match",
    "fans celebrated victory",
    "government economic policies",
    "economy grow year"
]

labels11 = ["sports", "sports", "economy", "economy"]

tfidf11 = TfidfVectorizer()
X11 = tfidf11.fit_transform(docs11)

model11 = MultinomialNB()
model11.fit(X11, labels11)

test11 = tfidf11.transform(["exciting football match"])
print(model11.predict(test11))



#12perform clustering of textual data using K-Means algorithm.

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

docs12 = [
    "football match fans",
    "team scored goals",
    "government education policies",
    "students education reforms"
]

tfidf12 = TfidfVectorizer()
X12 = tfidf12.fit_transform(docs12)

kmeans12 = KMeans(n_clusters=2)
kmeans12.fit(X12)

print(kmeans12.labels_)



#13Python program to build a text categorization model. SPAM 
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# a) Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Select required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# b) Text preprocessing function
def preprocess(text):
    text = text.lower()  # lowercase
    text = "".join([char for char in text if char not in string.punctuation])  # remove punctuation
    return text

df['message'] = df['message'].apply(preprocess)

# c) TF-IDF conversion
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['message'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# d) Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# e) Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


#14 N-gram tokenization 
from nltk.util import ngrams
from collections import Counter

text14 = "football team played exciting match won championship"

tokens14 = text14.split()

bigrams14 = list(ngrams(tokens14, 2))
trigrams14 = list(ngrams(tokens14, 3))

freq_bi14 = Counter(bigrams14)
freq_tri14 = Counter(trigrams14)

print(freq_bi14)
print(freq_tri14)

#15to analyze bigrams for sentiment context.
from nltk.util import ngrams
from collections import Counter

reviews15 = [
    "very good entertaining",
    "boring poorly written",
    "excellent impressive"
]

for review15 in reviews15:
    bigrams15 = list(ngrams(review15.split(), 2))
    freq15 = Counter(bigrams15)
    print(freq15)


#16Amazon Product Reviews Dataset Preprocess review text Visualize relationships using network graph. Extract bigrams
import pandas as pd

import re
import networkx as nx
import matplotlib.pyplot as plt
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Load dataset
df16 = pd.read(r"C:\Users\Manny\Desktop\label.docx")   # dataset file

# Step 2: Select text column
text16 = df16['review_text'].dropna()

# Step 3: Preprocessing (clean text)
clean16 = text16.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))

# Step 4: Extract bigrams
bigrams_list16 = []

for sentence in clean16:
    words16 = sentence.split()
    bigrams16 = list(ngrams(words16, 2))
    bigrams_list16.extend(bigrams16)

# Step 5: Count bigram frequency
freq16 = Counter(bigrams_list16)

# Display top bigrams
print(freq16.most_common(10))

# Step 6: Create network graph
G16 = nx.Graph()

for (w1, w2), freq in freq16.items():
    if freq > 5:   # filter important relationships
        G16.add_edge(w1, w2, weight=freq)

# Step 7: Draw graph
plt.figure(figsize=(8,6))
pos16 = nx.spring_layout(G16)

nx.draw(G16, pos16, with_labels=True, node_size=2000, font_size=10)
nx.draw_networkx_edge_labels(G16, pos16)

plt.show()



#17a Python program to perform topic modelling using Latent Dirichlet Allocation (LDA).
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs17 = [
    "football team won the championship match",
    "fans celebrated the football victory",
    "government announced new economic policies",
    "economy is growing steadily"
]

# Step 1: Convert text to document-term matrix
cv17 = CountVectorizer(stop_words='english')
X17 = cv17.fit_transform(docs17)

# Step 2: Apply LDA
lda17 = LatentDirichletAllocation(n_components=2, random_state=42)
lda17.fit(X17)

# Step 3: Display top words
words17 = cv17.get_feature_names_out()

for i, topic in enumerate(lda17.components_):
    print(f"Topic {i+1}:")
    print([words17[j] for j in topic.argsort()[-5:]])

#18Python program to demonstrate Hidden Markov Model (HMM) for Part-of-Speech tagging.
import numpy as np
import numpy as np

# Given sentence
sentence18 = "The young player scored a brilliant goal"
words18 = sentence18.lower().split()

# States (POS tags)
states18 = ["DET", "ADJ", "NOUN", "VERB"]

# Observations (words in sentence)
observations18 = words18

# Transition probabilities (example values)
transition18 = {
    "DET": {"ADJ": 0.6, "NOUN": 0.4},
    "ADJ": {"NOUN": 0.7, "ADJ": 0.3},
    "NOUN": {"VERB": 0.6, "NOUN": 0.4},
    "VERB": {"DET": 0.5, "NOUN": 0.5}
}

# Emission probabilities (word → POS likelihood)
emission18 = {
    "DET": {"the": 0.9, "a": 0.8},
    "ADJ": {"young": 0.8, "brilliant": 0.9},
    "NOUN": {"player": 0.9, "goal": 0.9},
    "VERB": {"scored": 0.9}
}

# Display

print("Transition Probabilities:", transition18)
print("Emission Probabilities:", emission18)


#15import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Step 1: Load dataset
df19 = pd.read_csv("news.csv")   # dataset file

# Step 2: Preprocess text
text19 = df19['text'].dropna()

clean_text19 = []
for t in text19:
    t = t.lower()                       # lowercase
    t = re.sub(r'[^a-z\s]', '', t)      # remove symbols
    clean_text19.append(t)

# Step 3: Convert to Document-Term Matrix
cv19 = CountVectorizer(stop_words='english')
X19 = cv19.fit_transform(clean_text19)

# Step 4: Apply LDA Model
lda19 = LatentDirichletAllocation(n_components=5, random_state=42)
lda19.fit(X19)

# Step 5: Display Top Words in Each Topic
words19 = cv19.get_feature_names_out()

for i, topic in enumerate(lda19.components_):
    print(f"\nTopic {i+1}:")
    top_words19 = [words19[j] for j in topic.argsort()[-10:]]
    print(top_words19)
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Step 1: Load dataset
df19 = pd.read_csv("news.csv")   # dataset file

# Step 2: Preprocess text
text19 = df19['text'].dropna()

clean_text19 = []
for t in text19:
    t = t.lower()                       # lowercase
    t = re.sub(r'[^a-z\s]', '', t)      # remove symbols
    clean_text19.append(t)

# Step 3: Convert to Document-Term Matrix
cv19 = CountVectorizer(stop_words='english')
X19 = cv19.fit_transform(clean_text19)

# Step 4: Apply LDA Model
lda19 = LatentDirichletAllocation(n_components=5, random_state=42)
lda19.fit(X19)

# Step 5: Display Top Words in Each Topic
words19 = cv19.get_feature_names_out()

for i, topic in enumerate(lda19.components_):
    print(f"\nTopic {i+1}:")
    top_words19 = [words19[j] for j in topic.argsort()[-10:]]
    print(top_words19)





