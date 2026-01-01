
import re, string
from collections import Counter
import numpy as np
import torch
import gensim
from cogworks_data.language import get_data_path
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    """
    Removes all punctuation from string
    input: corpus (str) --> output: corpus with punc removed
    """
    return punc_regex.sub('', corpus)

def process_doc(doc):
    ''' 
    Converts input to all lowercase, removes punctuation, and splits each word by space
    input: doc (str) -> returns (list)
    '''
    return strip_punc(doc).lower().split()

#3: Make a function that can embed any caption/query text (using GloVe-300 embeddings weighted by IDFs of words across captions). An individual word not in the GloVe or IDF vocabulary should yield an embedding vector of just zeros.
def load_glove():
    glove_input_file = r"C:\Users\sirid\OneDrive\Documents\BobaLoversCapstone\glove.6B\glove.6B.300d.txt"
    word2vec_output_file = r"C:\Users\sirid\OneDrive\Documents\BobaLoversCapstone\glove.6B\glove.6B.300d.txt.w2v"

    # Convert only if the .w2v file does not exist
    if not os.path.exists(word2vec_output_file):
        print("Converting GloVe format to Word2Vec format...")
        glove2word2vec(glove_input_file, word2vec_output_file)
        print("Conversion done!")

    print("Loading GloVe embeddings (this may take a while)...")
    glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    print("GloVe 300d embeddings loaded successfully!")
    return glove

def embed_text(text, glove):
    """
    Embeds list of words in text as GloVe embeddings.
    output: normalized embedding vector
    """
    word_list = process_doc(text)
    vectors = []
    for word in word_list:
        if word in glove:
            vectors.append(glove[word])
        else:
            vectors.append(np.zeros(300))
    return torch.tensor(vectors, dtype=torch.float32)