from time import time
import pandas as pd
import os
import re
import numpy as np
from pprint import pprint
import argparse
from lda2vec.nlppipe import Preprocessor



parser = argparse.ArgumentParser(description='Gather lda2vec topics')
parser.add_argument('--num_topics', type=int, default=10, help='number of topics')
parser.add_argument('--data_type', type=str, default='description', help='description or title')


args = parser.parse_args()


t0 = time()

print('INFO: done importing libraries and dataset in %0.3fs.' % (time() - t0))


t0 = time()
if args.data_type == 'title':

    clean_titles = []
    with open('datasets/parsed_full_titles.txt') as f:
        for line in f.readlines():
            clean_titles.append(line)

    titles = pd.DataFrame(clean_titles, columns=['processed_title'])


    # Where to save preprocessed data
    clean_data_dir = "data/clean_data"

    # Should we load pretrained embeddings from file
    load_embeds = True

    # Initialize a preprocessor
    P = Preprocessor(titles, "processed_title", max_features=30000, maxlen=10000, min_count=30)

    # Run the preprocessing on your dataframe
    t0 = time()

    print('INFO: beginning preprocesssing tokens from titles')

    P.preprocess()

    print('INFO: finished preprocessing tokens from titles in %0.3fs.' % (time() - t0))
else:
    clean_descriptions = []
    with open('datasets/parsed_full_descriptions.txt') as f:
        for line in f.readlines():
            clean_descriptions.append(line)

    descriptions = pd.DataFrame(clean_descriptions, columns=['processed_description'])

    # Where to save preprocessed data
    clean_data_dir = "data/clean_data"

    # Should we load pretrained embeddings from file
    load_embeds = True

    # Initialize a preprocessor
    P = Preprocessor(descriptions, "processed_description", max_features=30000, maxlen=10000, min_count=30)

    # Run the preprocessing on your dataframe
    t0 = time()

    print('INFO: beginning preprocesssing tokens from descriptions')

    P.preprocess()

    print('INFO: finished preprocessing tokens from descriptions in %0.3fs.' % (time() - t0))

# Load embeddings from file if we choose to do so
if load_embeds:
    # Load embedding matrix from file path - change path to where you saved them
    t0 = time()

    print('INFO: loading glove embeddings')
    embedding_matrix = P.load_glove("embeddings/glove.6B.300d.txt")

    print('INFO: finished loading glove embeddings in %0.3fs.' % (time() - t0))
else:
    embedding_matrix = None

# Save data to data_dir
P.save_data(clean_data_dir, embedding_matrix=embedding_matrix)

from lda2vec import utils, model

# print(tf.__version__)
# Path to preprocessed data
clean_data_dir = "data/clean_data"
# Whether or not to load saved embeddings file
load_embeds = True

# Load data from files
t0 = time()

print('INFO: loading preprocessed data')
(idx_to_word, word_to_idx, freqs, pivot_ids,
 target_ids, doc_ids, embed_matrix) = utils.load_preprocessed_data(clean_data_dir, load_embed_matrix=load_embeds)

print('INFO: finished loading preprocessed data in %0.3fs.' % (time() - t0))

# Number of unique documents
num_docs = doc_ids.max() + 1
# Number of unique words in vocabulary (int)
vocab_size = len(freqs)
# Embed layer dimension size
# If not loading embeds, change 128 to whatever size you want.
embed_size = embed_matrix.shape[1] if load_embeds else 128
# Number of topics to cluster into
num_topics = args.num_topics
# Amount of iterations over entire dataset
num_epochs = 300
# Batch size - Increase/decrease depending on memory usage
batch_size = 4096
# Epoch that we want to "switch on" LDA loss
switch_loss_epoch = 100
# Pretrained embeddings value
pretrained_embeddings = embed_matrix if load_embeds else None
# If True, save logdir, otherwise don't
save_graph = True

# Initialize the model
t0 = time()

print('INFO: initializing lda2vec model')
m = model(num_docs,
          vocab_size,
          num_topics,
          embedding_size=embed_size,
          pretrained_embeddings=pretrained_embeddings,
          freqs=freqs,
          batch_size = batch_size,
          save_graph_def=save_graph)

print('INFO: finished initializing lda2vec model in %0.3fs.' % (time() - t0))


# Train the model
t0 = time()

print('INFO: training lda2vec')
print('len(pivot_ids):', len(pivot_ids))
print('len(target_ids):', len(target_ids))
print('len(doc_ids)', len(doc_ids))
m.train(pivot_ids,
        target_ids,
        doc_ids,
        len(pivot_ids),
        num_epochs,
        idx_to_word=idx_to_word,
        switch_loss_epoch=switch_loss_epoch)

print('INFO: finished training lda2vec model in %0.3fs.' % (time() - t0))
