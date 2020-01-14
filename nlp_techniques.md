# Natural Language Processesing Techniques

Trying to extract skills from dataset of HITs. 

### Unstructured Text Papers
http://repository.fue.edu.eg/xmlui/bitstream/handle/123456789/3614/9166.pdf?sequence=1

@book{feldman2007text,
  title={The text mining handbook: advanced approaches in analyzing unstructured data},
  author={Feldman, Ronen and Sanger, James},
  year={2007},
  publisher={Cambridge university press}
}

Different approaches to analyzing unstructured data



@book{miner2012practical,
  title={Practical text mining and statistical analysis for non-structured text data applications},
  author={Miner, Gary and Elder IV, John and Fast, Andrew and Hill, Thomas and Nisbet, Robert and Delen, Dursun},
  year={2012},
  publisher={Academic Press}
}

Stat analysis for unstructed text. 

https://www.researchgate.net/publication/224266411_Analysis_and_evaluation_of_unstructured_data_Text_mining_versus_natural_language_processing

Text mining vs. NLP
@inproceedings{gharehchopogh2011analysis,
  title={Analysis and evaluation of unstructured data: text mining versus natural language processing},
  author={Gharehchopogh, Farhad Soleimanian and Khalifelu, Zeinab Abbasi},
  booktitle={2011 5th International Conference on Application of Information and Communication Technologies (AICT)},
  pages={1--4},
  year={2011},
  organization={IEEE}
}

Breakdown of Text Mining
https://thesai.org/Downloads/Volume7No11/Paper_53-Text_Mining_Techniques_Applications_and_Issues.pdf
## Questions for Dr. Yue
- Would you like me to focus on creating a skill extraction engine?
- Or would you like me to focus on qualitative analysis of the text?
- Clustering of the HITs, would be valuable for classifying the type of HIT.
- Create a summarization engine for HITs



### ELMo
https://allennlp.org/elmo
"deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy)."

Can be added to existing models and significantly improve the state of the art.

### Transfer Learning for Text Classificication
https://arxiv.org/abs/1801.06146
"We propose Universal Language Model Fine-tuning (ULMFiT), an effective transfer learning method that can be applied to any task in NLP, and introduce techniques that are key for fine-tuning a language model. Our method significantly outperforms the state-of-the-art on six text classification tasks, reducing the error by 18-24% on the majority of datasets. Furthermore, with only 100 labeled examples, it matches the performance of training from scratch on 100x more data. We open-source our
pretrained models and code."

### Word Embeddings
https://www.tensorflow.org/tutorials/text/word_embeddings

### Topic Classification
https://developers.google.com/machine-learning/guides/text-classification/step-2-51

### Sentiment Analysis
word2vec
