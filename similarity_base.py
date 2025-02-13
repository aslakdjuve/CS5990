# -------------------------------------------------------------------------
# AUTHOR: Aslak Djuve
# FILENAME: Question 8
# SPECIFICATION: 
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 30 min
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append (row)
         print(row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection without applying any transformations, using
# the spaces as your character delimiter.
#--> add your Python code here
docTermMatrix = []
uniqueWords = set()

# First pass - collect all unique words
for doc in documents:
    words = doc[1].split() # doc[1] contains the text
    uniqueWords.update(words)

uniqueWords = list(uniqueWords) # Convert to list for consistent indexing

# Second pass - build binary document-term matrix
for doc in documents:
    vector = [0] * len(uniqueWords)
    words = doc[1].split()
    for word in words:
        if word in uniqueWords:
            vector[uniqueWords.index(word)] = 1
    docTermMatrix.append(vector)

# Compare pairwise cosine similarities
maxSimilarity = 0
doc1Index = 0
doc2Index = 0

for i in range(len(docTermMatrix)):
    for j in range(i + 1, len(docTermMatrix)):
        similarity = cosine_similarity([docTermMatrix[i]], [docTermMatrix[j]])[0][0]
        if similarity > maxSimilarity:
            maxSimilarity = similarity
            doc1Index = i
            doc2Index = j

# Print results
print(f"The most similar documents are document {documents[doc1Index][0]} and document {documents[doc2Index][0]} with cosine similarity = {maxSimilarity:.4f}")

