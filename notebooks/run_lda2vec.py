from time import time
import pandas as pd
import os
import re
from pprint import pprint


# import gensim
# from gensim.utils import simple_preprocess
# import gensim.corpora as corpora
# from gensim.models import CoherenceModel
#
# from matplotlib.ticker import FuncFormatter
# from matplotlib import pyplot as plt


import pyLDAvis
import pyLDAvis.gensim
import pickle

# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords

t0 = time()

print('INFO: done importing libraries and dataset in %0.3fs.' % (time() - t0))


t0 = time()

hits = pd.read_json('datasets/clean_hits_full_week.json', lines=True)
hits.head()

print('INFO: done reading in dataset to pandas dataframe in %0.3fs.' % (time() - t0))



t0 = time()

hits = hits.drop(columns=['_id', 'hit_set_id', 'requester_id','requester_name', 'assignment_duration_in_seconds', 'creation_time', 'assignable_hits_count', 'latest_expiration_time', 'caller_meets_requirements', 'caller_meets_preview_requirements', 'last_updated_time', 'monetary_reward', 'accept_project_task_url', 'requester_url', 'project_tasks_url', 'project_requirements', 'requesterInfo'], axis=1)
hits.head()

print('INFO: done columns from dataframe in %0.3fs.' % (time() - t0))

# removes all punctuation from the description and title if any
t0 = time()

hits['processed_description'] = hits['description'].map(lambda d : re.sub('[,.!?]', '', d))
hits['processed_title'] = hits['title'].map(lambda t : re.sub('[,.!?]', '', t))

print('INFO: done removing punctuation from title and description in %0.3fs.' % (time() - t0))


# cleans dataframe by converting all characters to lowercase and removing non-english characters
t0 = time()
print('INFO: beginning to clean dataframe')

# constructs allowable english chars
def clean_dat(chunk):
    # allowable_chars = [ chr(i) for i in range(128) ]
    # allowable_chars = set(allowable_chars)

    # Read stopwords
    with open('datasets/stops.txt', 'r') as f:
        stops = f.read().split('\n')

    return ' '.join([ w for w in chunk.split() if w not in set(stops) and not w.isnumeric()])

# converts to low caps
hits['processed_description'] = hits['processed_description'].map(lambda x: x.lower())

print('processed_description shape before dropping empty descriptions',hits['processed_description'].shape)

# removes non allowable characters
hits['processed_description'] = hits['processed_description'].map(lambda x: clean_dat(x))

descriptions = pd.DataFrame(data = hits['processed_description'])
# drops row if processed_description is empty
#hits.dropna(subset=['processed_description'])
descriptions.dropna()
descriptions.drop_duplicates(keep=False, inplace=True)
# hits.drop_duplicates(subset=['processed_description'])
print('processed_description shape after dropping empty descriptions', descriptions.shape)

print('INFO: finished removing non english characters in %0.3fs' % (time() - t0))

hits['processed_title'] = hits['processed_title'].map(lambda x: x.lower())





# print out the first couple processed descriptions
#hits['processed_description'].head()
descriptions.head()



from lda2vec.nlppipe import Preprocessor

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
num_topics = 10
# Amount of iterations over entire dataset
num_epochs = 250
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

