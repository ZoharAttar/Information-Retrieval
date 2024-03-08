from flask import Flask, request, jsonify
from BackEnd import *
from google.cloud import storage
import pickle
from nltk.corpus import stopwords
import re
import concurrent.futures


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# text

bucket_name = '332396282'

client = storage.Client()
bucket = client.get_bucket(bucket_name)

blob = bucket.blob('final/text_postings/index.pkl')
pkl = blob.download_as_string()
index_text = pickle.loads(pkl)

blob = bucket.blob('corpus_len.pkl')
pkl = blob.download_as_string()
corpus_len = pickle.loads(pkl)

blob = bucket.blob('final/text_normalization.pkl')
pkl = blob.download_as_string()
doc_normalization = pickle.loads(pkl)

blob = bucket.blob('final/idf_text.pkl')
pkl = blob.download_as_string()
idf_text = pickle.loads(pkl)

blob = bucket.blob('final/text_len_dict.pkl')
pkl = blob.download_as_string()
text_len_dict = pickle.loads(pkl)

words_text = list(index_text.df.keys())

blob = bucket.blob('final/doc_title.pkl')
pkl = blob.download_as_string()
doc_title = pickle.loads(pkl)

# title

blob = bucket.blob('title_InvertedIndex/title_InvertedIndex.pkl')
contents = blob.download_as_bytes()
index_title = pickle.loads(contents)

blob = bucket.blob('final/titles_normalization.pkl')
pkl = blob.download_as_string()
titles_normalization = pickle.loads(pkl)

blob = bucket.blob('final/idf_title.pkl')
pkl = blob.download_as_string()
idf_title = pickle.loads(pkl)

blob = bucket.blob('final/title_len_dict.pkl')
pkl = blob.download_as_string()
title_len_dict = pickle.loads(pkl)

words_title = list(index_title.df.keys())

# page rank
blob = bucket.blob('page_rank.pkl')
pkl = blob.download_as_string()
page_rank = pickle.loads(pkl)

# BM25

blob = bucket.blob('avg_dl_dict.pkl')
pkl = blob.download_as_string()
avg_dl_dict = pickle.loads(pkl)

blob = bucket.blob('text_term_idf_bm25.pkl')
pkl = blob.download_as_string()
text_term_idf_bm25 = pickle.loads(pkl)

blob = bucket.blob('title_term_idf_bm25.pkl')
pkl = blob.download_as_string()
title_term_idf_bm25 = pickle.loads(pkl)

blob = bucket.blob('anchor_term_idf_bm25.pkl')
pkl = blob.download_as_string()
anchor_term_idf_bm25 = pickle.loads(pkl)


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    tokenized_query = tokenize(query)
    if len(query) == 1:
        cs_docs_text = get_candidate_documents_and_scores_bm25_for_len1(tokenized_query, words_text, index_text,
                                                                        text_term_idf_bm25,
                                                                        text_len_dict,
                                                                        'final/text_postings', avg_dl_dict['text'],
                                                                        bucket_name)
        top_n_docs_text = get_top_n(cs_docs_text, 70)
        cs_docs_title = get_candidate_documents_and_scores_bm25_for_len1(tokenized_query, words_title, index_title,
                                                                         title_term_idf_bm25,
                                                                         title_len_dict, '', avg_dl_dict['title'],
                                                                         bucket_name)
        top_n_docs_title = get_top_n(cs_docs_title, 70)
        merged = merge([top_n_docs_text, top_n_docs_title], page_rank)
        top_n = get_top_n(merged)
        res = get_titles(top_n, doc_title)
        return jsonify(res)

    top_n_docs_title = search_title(tokenized_query)
    top_n_docs_text = search_body(tokenized_query)

    merged = merge([top_n_docs_text, top_n_docs_title], page_rank)
    top_n = get_top_n(merged)
    res = get_titles(top_n, doc_title)
    return jsonify(res)


@app.route("/search_body")
def search_body(tokenized_query):
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    cs_docs_text = get_candidate_documents_and_scores_bm25(tokenized_query, words_text, index_text,
                                                           text_term_idf_bm25,
                                                           text_len_dict,
                                                           'final/text_postings', avg_dl_dict['text'], bucket_name)
    top_n_docs_text = get_top_n(cs_docs_text, 70)
    return top_n_docs_text


@app.route("/search_title")
def search_title(tokenized_query):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    cs_docs_title = get_candidate_documents_and_scores_bm25(tokenized_query, words_title, index_title,
                                                            title_term_idf_bm25,
                                                            title_len_dict, '', avg_dl_dict['title'], bucket_name)
    top_n_docs_title = get_top_n(cs_docs_title, 70)
    return top_n_docs_title


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
