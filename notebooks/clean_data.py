from time import time
import pandas as pd
import os
import re
import numpy as np
from pprint import pprint

t0 = time()

print('INFO: done importing libraries and dataset in %0.3fs.' % (time() - t0))


t0 = time()

hits = pd.read_json('datasets/20200126-20200312-hits.json', lines=True)
hits.head()

print('INFO: done reading in dataset to pandas dataframe in %0.3fs.' % (time() - t0))



t0 = time()

hits = hits.drop(columns=['_id', 'hit_set_id', 'requester_id','requester_name', 'assignment_duration_in_seconds', 'creation_time', 'assignable_hits_count', 'latest_expiration_time', 'caller_meets_requirements', 'caller_meets_preview_requirements', 'last_updated_time', 'monetary_reward', 'accept_project_task_url', 'requester_url', 'project_tasks_url', 'project_requirements', 'requesterInfo'], axis=1)
hits.head()

print('INFO: done columns from dataframe in %0.3fs.' % (time() - t0))

# removes all punctuation from the description and title if any
t0 = time()

hits['processed_description'] = hits['description'].map(lambda d : re.sub('[,.$()@#%&~!?]', '', d))
hits['processed_title'] = hits['title'].map(lambda t : re.sub('[,.$()@#%&~!?]', '', t))

print('INFO: done removing punctuation from title and description in %0.3fs.' % (time() - t0))


# cleans dataframe by converting all characters to lowercase and removing non-english characters
t0 = time()

def clean_dat(chunk):
    # Read stopwords
    with open('datasets/stops.txt', 'r') as f:
        stops = f.read().split('\n')

    return ' '.join([ w for w in chunk.split() if w not in set(stops) and not w.isnumeric()])

# converts to low caps
hits['processed_description'] = hits['processed_description'].map(lambda x: x.lower())

hits['processed_title'] = hits['processed_title'].map(lambda x: x.lower())

print('INFO: removing stopwords, duplicates, and numbers')

print('INFO: processed_description shape before dropping empty descriptions',hits['processed_description'].shape[0])
print('INFO: processed_title shape before dropping stop words and number', hits['processed_title'].shape[0])

t0 = time()

# removes non allowable characters
hits['processed_description'] = hits['processed_description'].map(lambda x: clean_dat(x))
hits['processed_title'] = hits['processed_title'].map(lambda x: clean_dat(x))

# create a descriptions data frame
descriptions = pd.DataFrame(data = hits['processed_description'])
titles = pd.DataFrame(data = hits['processed_title'])

# drops row if processed_description is empty
descriptions.dropna()
titles.dropna()

descriptions.drop_duplicates(keep=False, inplace=True)
titles.drop_duplicates(keep=False, inplace=True)

print('INFO: processed_description shape after removing stopwords, duplicates, and numbers', descriptions.shape[0])

print('INFO: processed_title shape after removing stopwords, duplicates, and numbers', titles.shape[0])

print('INFO: finished removing stopwords, duplicates, and numbers in %0.3fs' % (time() - t0))

# print out the first couple processed descriptions
t0 = time()
print('INFO: loading descriptions into text file')

descriptions.to_csv(r'datasets/parsed_full_descriptions.txt', header=None, index=None, sep=' ', mode='a')
print('INFO: finished loading descriptions into text file in %0.3fs' % (time() - t0))


# print out the first couple processed titles
t0 = time()
print('INFO: loading titles into text file')

titles.to_csv(r'datasets/parsed_full_titles.txt', header=None, index=None, sep=' ', mode='a')
print('INFO: finished loading titles into text file in %0.3fs' % (time() - t0))

