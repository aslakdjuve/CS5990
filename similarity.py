# -------------------------------------------------------------------------
# AUTHOR: Aslak Djuve
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: how long it took you to complete the assignment
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
         #print(row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection without applying any transformations, using
# the spaces as your character delimiter.
#--> add your Python code here
docTermMatrix = []
def unique_words(documents):
  """
  Iterating through the documents to return a vector with all the unique words

  """
  word_vector = []
  for line in documents:
      temp = line[1].split(" ")
      for i in temp:
        if i not in word_vector:
            word_vector.append(i)
  return(word_vector)


word_vector = unique_words(documents)

def create_term_matrix(documents, docTermMatrix, word_vector):
  for line in documents:
    line = line[1]
    temp = []
    for i in word_vector:
      if i in line:
        temp.append(1)
      else:
         temp.append(0)
    docTermMatrix.append(temp)
  return docTermMatrix
    
docTermMatrix = create_term_matrix(documents, docTermMatrix, word_vector)
  
print(docTermMatrix[0])
 
#print((len(docTermMatrix))) #401, correct number of vectors
#print(docTermMatrix)

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here
similarity_matrix = cosine_similarity(docTermMatrix)
max_sim = -1.0
best_i = -1
best_j = -1

for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i][j] > max_sim:
            max_sim = similarity_matrix[i][j]
            best_i = i
            best_j = j


# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
print(f"The most similar documents are document {documents[best_i][0]} and document {documents[best_j][0]} with cosine similarity = {max_sim:.4f}.")


