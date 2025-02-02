from time import time
import pandas as pd
import os
import re
from pprint import pprint
import argparse

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
  print('INFO: generating lda model')

  lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=args.num_topics,
                                        random_state=100,
                                        chunksize=100,
                                        passes=13,
                                        per_word_topics=True)
  return lda_model

def print_lda(lda):
    print('INFO: printing topics in print_lda')
    pprint(lda.print_topics(num_topics=args.num_topics, num_words=10))

def get_topics(lda):
    print('INFO: printing topics in print_topics')
    lda_topics = lda.show_topics(num_topics=args.num_topics, num_words=10, formatted=False)

    # pprint(lda_topics)
    topics = []
    for i, topic in enumerate(lda_topics):
        print('topic', i, topic[1])

        topic_words = []
        for word_tuple in topic[1]:
           word = word_tuple[0]
           topic_words.append(word)
        topics.append(topic_words)

    print(topics)

    return topics

def write_topics(topics, filename):
    with open(filename, 'w') as f:
        for topic in topics:
            f.write(', '.join(topic) + '\n')

t0 = time()
if args.data_type == 'title':

    print('INFO: generating lda with ', args.num_topics, ' topics')

    clean_titles = []
    with open('datasets/parsed_full_titles.txt') as f:
        t0 = time()

        title_tokens = []
        for line in f.readlines():
            clean_titles.append(line)
            title_tokens = list(sent_to_words(clean_titles))

        print('INFO: finished tokenizing titles %0.3fs.' % (time() - t0))

    # generate id2word dictionary of title tokens
    t0 = time()

    print('INFO: generating title id2word dict')

    title_id2word = corpora.Dictionary(title_tokens)

    print('INFO: finished generating id2word dict %0.3fs.' % (time() - t0))

    # generate corpus from title tokens
    t0 = time()

    print('INFO: generating title corpus')

    title_corpus = [title_id2word.doc2bow(tok) for tok in title_tokens]

    print('INFO: done generating title corpus in %0.3fs.' % (time() - t0))
    #descriptions = pd.DataFrame(clean_descriptions, columns=['processed_description'])

    # generate title bigram LDA
    t0 = time()

    print('INFO: generating LDA')

    title_lda = generate_lda(title_corpus, title_id2word, args.num_topics)

    print('INFO: done generating title LDA in %0.3fs.' % (time() - t0))

    # print title lda topics
    t0 = time()

    print('INFO: printing lda')

    print_lda(title_lda)

    print('INFO: finished printing topics in print_lda in %0.3fs.' % (time() -t0))

    t0 = time()

    title_topics = get_topics(title_lda)

    print('INFO: done printing title LDA topics from get_topics in %0.3fs.' % (time() - t0))

    t0 = time()

    title_file_name = 'results/lda/titles/lda-title-' + str(args.num_topics) + 'topics.txt'

    print('INFO: writing topics to', title_file_name)

    write_topics(title_topics, title_file_name)

    print('INFO: finished writing topics to', title_file_name, ' in %0.3fs.' % (time() - t0))
else:
    print('INFO: generating lda with ', args.num_topics, ' topics')

    clean_descriptions = []
    with open('datasets/parsed_full_descriptions.txt') as f:
        t0 = time()

        description_tokens = []
        for line in f.readlines():
            clean_descriptions.append(line)
            description_tokens = list(sent_to_words(clean_descriptions))

        print('INFO: finished tokenizing descriptions %0.3fs.' % (time() - t0))

    # generate id2word dictionary of description tokens
    t0 = time()

    print('INFO: generating description id2word dict')

    description_id2word = corpora.Dictionary(description_tokens)

    print('INFO: finished generating id2word dict %0.3fs.' % (time() - t0))

    # generate corpus from description tokens
    t0 = time()

    print('INFO: generating description corpus')

    description_corpus = [description_id2word.doc2bow(tok) for tok in description_tokens]

    print('INFO: done generating description corpus in %0.3fs.' % (time() - t0))
    #descriptions = pd.DataFrame(clean_descriptions, columns=['processed_description'])

    # generate description bigram LDA
    t0 = time()

    print('INFO: generating LDA')

    description_lda = generate_lda(description_corpus, description_id2word, args.num_topics)

    print('INFO: done generating description LDA in %0.3fs.' % (time() - t0))

    # print description lda topics
    t0 = time()

    print('INFO: printing lda')

    print_lda(description_lda)

    print('INFO: finished printing topics in print_lda in %0.3fs.' % (time() -t0))

    t0 = time()

    description_topics = get_topics(description_lda)

    print('INFO: done printing description LDA topics from get_topics in %0.3fs.' % (time() - t0))

    t0 = time()

    description_file_name = 'results/lda/descriptions/lda-description-' + str(args.num_topics) + 'topics.txt'

    print('INFO: writing topics to', description_file_name)

    write_topics(description_topics, description_file_name)

    print('INFO: finished writing topics to', description_file_name, ' in %0.3fs.' % (time() - t0))

