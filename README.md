# Information-Retrieval
Building a search engine for english wikipedia files

## Overview

This project is aimed at building a search engine capable of efficiently retrieving relevant documents based on user queries. The search engine utilizes various techniques including tokenization, indexing, and ranking algorithms to provide accurate search results. Our serch engine uses BM25 scoring method to determine the most relevant documents.

## Code Description

The project consists of several Python modules:

- `search_frontend.py`: Implements the main functionality of the search engine including tokenization, indexing, and retrieval algorithms.
- `inverted_index_gcp.py`: Implements the inverted Index class.
- `BackEnd.py`: Implements functions for processing user queries and retrieving relevant documents with different methods.
- `final_notebook.py`: Building the indices and dictionaries needed for the search engine and storing them in a GCP bucket.

## Functions

# get_candidate_documents_tfidf :
Retrieves candidate documents and their scores based on the input query. It calculates the TF-IDF score for each term in the query and matches them with the indexed documents.

# query_tfidf_dic:
Generates a dictionary containing the TF-IDF scores for each term in the query. It calculates the TF-IDF score for each term in the query based on the IDF values provided.

# query_tf_dic:
Generates a dictionary containing the TF scores for each term in the query. It calculates the TF score for each term in the query based on the term frequency.

# get_candidate_docs_cosine_sim:
Computes the cosine similarity between the query and candidate documents. It normalizes the TF-IDF scores for both the query and documents and calculates the cosine similarity.

# get_top_n:
Retrieves the top N entries from a similarity dictionary sorted by their scores. It sorts the dictionary based on the scores and returns the top N entries. Default value for N is 30.

# get_titles:
Retrieves the titles of the documents corresponding to the top N entries. It matches the document IDs in the top N list with their titles and returns a list of tuples containing document IDs and titles.

# merge:
Merges the scores from Title, Text and PageRank for each document. It combines the scores using weighted averaging and returns a dictionary containing the merged scores for each document.

# get_candidate_documents_and_scores_bm25_for_len1:
Retrieves candidate documents and their scores using the BM25 algorithm. It calculates the BM25 score for queries containing consisted of 1 word and matches them with the indexed documents.

# get_candidate_documents_and_scores_bm25:
Retrieves candidate documents and their scores using the BM25 algorithm with multithreading support. It processes each term in the query using a separate thread to improve performance.

# process_term: 
Processes a single term from the query to calculate its contribution to the BM25 score. It retrieves the posting list for the term from the index and updates the scores for candidate documents using a lock to ensure thread safety.

## Contributors

- [Zohar Attar](https://github.com/zoharattar)
- [Gaby Levis]

