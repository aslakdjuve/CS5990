# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: similarity.py
# SPECIFICATION: Find and output the two most similar documents from cleaned_documents.csv based on cosine similarity
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: how long it took you to complete the assignment
# -------------------------------------------------------------------------

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
# pandas, or other sklearn modules.
# You have to work here only with standard dictionaries, lists, and arrays

# Importing required libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

# Read documents from CSV file
documents = []
with open('cleaned_documents.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0 and row:  # Skipping the header and ensuring the row is not empty
            documents.append(row[0].strip())  # Stripping spaces to avoid formatting issues

# Ensure there are documents to process
if not documents:
    print("No documents found in the file.")
    exit()

# Extract unique words to build vocabulary
vocabulary = set()
for doc in documents:
    words = doc.split()  # Splitting by spaces
    vocabulary.update(words)
vocabulary = sorted(vocabulary)  # Sorting to maintain consistency

# Build the document-term matrix using binary encoding
doc_term_matrix = []
for doc in documents:
    word_presence = [1 if word in doc.split() else 0 for word in vocabulary]
    doc_term_matrix.append(word_presence)

# Compute pairwise cosine similarities
max_similarity = -1  # Setting to -1 to ensure we capture valid similarities
most_similar_docs = (-1, -1)
for i in range(len(doc_term_matrix)):
    for j in range(i + 1, len(doc_term_matrix)):
        similarity = cosine_similarity([doc_term_matrix[i]], [doc_term_matrix[j]])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_docs = (i + 1, j + 1)  # Assuming documents are 1-indexed

# Ensure valid output
if most_similar_docs == (-1, -1):
    print("No valid similarities found.")
else:
    print(f"The most similar documents are document {most_similar_docs[0]} and document {most_similar_docs[1]} with cosine similarity = {max_similarity:.4f}")