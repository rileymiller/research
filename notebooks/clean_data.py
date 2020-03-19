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


# import pyLDAvis
# import pyLDAvis.gensim
# import pickle

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
    allowable_chars = [ chr(i) for i in range(128) ]
    allowable_chars = set(allowable_chars)


    clean_chunk = ''
    for word in chunk:
        isClean = True
        for char in word:
            if char not in allowable_chars:
                isClean = False
                break
        if isClean:
            clean_chunk = clean_chunk + word

    return clean_chunk

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
