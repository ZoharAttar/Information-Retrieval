import concurrent.futures
import re
from collections import Counter, defaultdict
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
import hashlib
import threading

nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


def token2bucket_id(token):
    return int(_hash(token), 16) % NUM_BUCKETS


def tokenize(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = [token for token in tokens if token not in all_stopwords]
    return tokens


def get_candidate_documents_tfidf(query_to_search, words, index, idf, docs_len, blob, bucket_name):
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = index.read_a_posting_list(blob, term, bucket_name)
            for tup in list_of_doc:
                doc_id = tup[0]
                tf = tup[1] / docs_len[doc_id]
                idf1 = idf[term]
                candidates[(doc_id, term)] = (tf * idf1)

    return candidates


def query_tfidf_dic(query_to_search, words, idf):
    Q = {}

    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in words:  # avoid terms that do not appear in the index.
            tf = (counter[token] / len(query_to_search))  # term frequency divded by the length of the query
            idf1 = idf[token]
            Q[token] = (tf * idf1)

    return Q


def query_tf_dic(query_to_search, words):
    Q = {}

    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in words:  # avoid terms that do not appear in the index.
            tf = np.divide(counter[token], len(query_to_search))  # term frequency divded by the length of the query
            Q[token] = tf

    return Q


def get_candidate_docs_cosine_sim(query, docs, doc_norm):
    query_norm = np.linalg.norm(np.array([float(x) for x in list(query.values())]))
    cossim = {}

    for tpl in docs.keys():
        if tpl[0] in cossim.keys():
            cossim[tpl[0]] += (docs[tpl] * query[tpl[1]])
    else:
        cossim[tpl[0]] = (docs[tpl] * query[tpl[1]])

    for key in cossim.keys():
        cossim[key] = (cossim[key] / (doc_norm[key] * query_norm))

    return cossim


def get_top_n(sim_dict, N=30):
    ret = {}
    ret = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    return dict(ret[:N])


def get_titles(top_n_list, doc_titles):
    ret = []
    for key in top_n_list:
        if key in doc_titles:
            ret.append((str(key), doc_titles[key]))
        else:
            ret.append((str(key), ''))

    return ret


def merge(c_lst, page_rank):
    ret_dict = defaultdict(float)
    for c in range(len(c_lst)):
        for doc_id, score in c_lst[c].items():
            if c == 0:
                weighted_score = score * 0.15
                ret_dict[doc_id] += weighted_score
                ret_dict[doc_id] += np.log10(page_rank[doc_id])
            else:
                weighted_score = score * 0.85
                ret_dict[doc_id] += weighted_score
                ret_dict[doc_id] += np.log10(page_rank[doc_id])
    return ret_dict


def get_candidate_documents_and_scores_bm25_for_len1(query_to_search, words, index, idf_dict, docs_len, blob, avgdl,
                                                     bucket_name, b=0.75, k1=1.5):
    candidates = defaultdict(float)
    B_dict = defaultdict(float)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = index.read_a_posting_list(blob, term, bucket_name)
            for tup in list_of_doc:
                doc_id = tup[0]
                tf = tup[1]
                idf = idf_dict[term]
                if doc_id in B_dict:
                    B = B_dict[doc_id]
                else:
                    B = (1 - b + (b * (docs_len[doc_id] / avgdl)))
                numerator = (idf * (tf * (k1 + 1)))
                denominator = tf + (k1 * B)
                candidates[doc_id] += (numerator / denominator)

    return candidates


candidates_lock = threading.Lock()


def process_term(term, words, index, idf_dict, docs_len, blob, avgdl, bucket_name, candidates,
                 b=0.75, k1=1.2):
    B_dict = defaultdict(float)
    if term in words:
        list_of_doc = index.read_a_posting_list(blob, term, bucket_name)
        for tup in list_of_doc:
            doc_id = tup[0]
            tf = tup[1]
            idf = idf_dict[term]
            if doc_id in B_dict:
                B = B_dict[doc_id]
            else:
                B = (1 - b + (b * (docs_len[doc_id] / avgdl)))

            with candidates_lock:
                candidates[doc_id] += ((idf * (tf * (k1 + 1))) / (tf + (k1 * B)))


def get_candidate_documents_and_scores_bm25(query_to_search, words, index, idf_dict, docs_len, blob, avgdl, bucket_name,
                                            b=0.75, k1=1.2):
    candidates = defaultdict(float)
    # tf_query = query_tf_dic(query_to_search, words)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for word in np.unique(query_to_search):
            executor.submit(process_term, word, words, index, idf_dict, docs_len, blob, avgdl,
                            bucket_name, candidates,
                            b, k1)

    return candidates
