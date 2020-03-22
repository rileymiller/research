from time import time
import pandas as pd
import os
import re
from pprint import pprint


import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models import CoherenceModel

parser = argparse.ArgumentParser(description='Gather lda2vec topics')
parser.add_argument('--num_topics', type=int, default=10, help='number of topics')
parser.add_argument('--data_type', type=str, default='description', help='description or title')


args = parser.parse_args()


t0 = time()

print('INFO: done importing libraries and dataset in %0.3fs.' % (time() - t0))


# function to tokenize the unstructured text
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def generate_lda(corpus, id2word, num_topics):
  lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=20,
                                        random_state=100,
                                        chunksize=100,
                                        passes=13,
                                        per_word_topics=True)
  return lda_model

def print_lda(lda):
  pprint(lda.print_topics(num_topics=args.num_topics, num_words=10))


t0 = time()
if args.data_type == 'title':

    clean_titles = []
    with open('datasets/parsed_full_titles.txt') as f:
        for line in f.readlines():
            clean_titles.append(line)

    titles = pd.DataFrame(clean_titles, columns=['processed_title'])
else:

    clean_descriptions = []
    with open('datasets/parsed_full_descriptions.txt') as f:
        for line in f.readlines():
            clean_descriptions.append(line)
            description_tokens = list(sent_to_words(description_data))

            t0 = time()

    # generate id2word dictionary of description tokens
    description_id2word = corpora.Dictionary(description_tokens)

    # generate corpus from description tokens
    description_corpus = [description_id2word.doc2bow(tok) for tok in description_tokens]

    print('INFO: done generating description corpus in %0.3fs.' % (time() - t0))
    #descriptions = pd.DataFrame(clean_descriptions, columns=['processed_description'])

    # generate description bigram LDA
    t0 = time()

    description_lda = generate_lda(description_corpus, description_id2word, args.num_topics)

    print('INFO: done generating description bigram LDA in %0.3fs.' % (time() - t0))

    # print description lda topics
    t0 = time()

    print_lda(description_lda)

    print('INFO: done printing description LDA topics in %0.3fs.' % (time() - t0))
