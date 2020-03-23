from time import time
import pandas as pd
import os
import re
from pprint import pprint

from getpass import getpass
from html import unescape

import json
from bs4 import BeautifulSoup

import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models import CoherenceModel

preview_src_cnt = 0
preview_no_src_cnt = 0
previews = []

def clean_dat(chunk):
    # Read stopwords
    with open('datasets/stops.txt', 'r') as f:
        stops = f.read().split('\n')

    return ' '.join([ w for w in chunk.split() if w not in set(stops)])

for line in open('datasets/20200126-20200312-preview.json'):
      preview = json.loads(line)

      if 'page_src' in preview:
        preview_src_cnt += 1
        page_src = preview['page_src']
        soup = BeautifulSoup(page_src, 'html.parser')
        # print(soup.find_all('script'))

        # Clear every script tag
        for tag in soup.find_all('script'):
            tag.clear()

        # Clear every style tag
        for tag in soup.find_all('style'):
            tag.clear()

        # print(soup.get_text())
        clean_src = re.sub("<.*?>", "", soup.get_text())
        clean_src = re.sub("\n", " ", clean_src)
        previews.append(clean_src)
      else:
        preview_no_src_cnt += 1

print('INFO: done parsing preview dataset, finished in %0.3fs.' % (time() - t0))
print('INFO: total previews: ', (preview_src_cnt + preview_no_src_cnt), ' Previews with page src: ', preview_src_cnt, ' Preview w/o page src: ', preview_no_src_cnt)


preview_df = pd.DataFrame(previews, columns=['processed_previews'])
preview_df.head()

preview_df['processed_previews'] = preview_df['processed_previews'].map(lambda d : re.sub('[,.$()@#%&~!?]', '', d))
hits['processed_previews'] = hits['processed_previews'].str.replace('\W', ' ')
hits['processed_previews'] = hits['processed_previews'].map(lambda d : re.sub('\d', '', d))
hits['processed_previews'] = hits['processed_previews'].str.replace('\s+', ' ')
preview_df['processed_previews'] = preview_df['processed_previews'].map(lambda d : d.lower())

print('INFO: processed_preview shape before removing stop words and dropping empty previews', preview_df['processed_previews'].shape[0])

preview_df['processed_previews'] = preview_df['processed_previews'].map(lambda d : clean_dat(d))

preview_df['processed_previews'] = preview_df['processed_previews'].map(lambda d : re.sub('"', '', d))
preview_df['processed_previews'] = preview_df['processed_previews'].map(lambda d : re.sub("''", '', d))

nan_value = float("NaN")
preview_df.replace("", nan_value, inplace=True)

preview_df.dropna(subset = ['processed_previews'], inplace=True)


preview_df['processed_previews'].drop_duplicates(keep=False, inplace=True)


print('INFO: processed_preview shape after removing stop words and dropping empty previews', preview_df['processed_previews'].shape[0])




# print out the first couple processed descriptions
t0 = time()

print('INFO: loading previews into text file')
preview_df['processed_previews'].to_csv(r'datasets/parsed_full_previews.txt', header=None, index=None, sep=' ', mode='a')

print('INFO: finished loading previews into text file in %0.3fs' % (time() - t0))

print(new_preview)
