from time import time
import pandas as pd
import os
import re
from pprint import pprint

from getpass import getpass
from html import unescape


import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models import CoherenceModel

from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt


import pyLDAvis
import pyLDAvis.gensim
import pickle

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import json
from bs4 import BeautifulSoup


t0 = time()


print('INFO: done importing libraries in %0.3fs.' % (time() - t0))

# convert preview.json file into dictionary, populate previews list

preview_src_cnt = 0
preview_no_src_cnt = 0
previews = []
count = 0
for line in open('datasets/20200126-20200312-preview.json'):
      preview = json.loads(line)

      count += 1

      if count == 25:
        break

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
        previews.append(clean_src)
      else:
        preview_no_src_cnt += 1

print('INFO: done parsing preview dataset, finished in %0.3fs.' % (time() - t0))
print('INFO: total previews: ', (preview_src_cnt + preview_no_src_cnt), ' Previews with page src: ', preview_src_cnt, ' Preview w/o page src: ', preview_no_src_cnt)

preview = pd.DataFrame(previews, columns=['processed_previews'])


