import nltk
words = set(nltk.corpus.words.words())

sent = "Io andiamo to the beach with my amico."
" ".join(w for w in nltk.wordpunct_tokenize(sent) \
         if w.lower() in words or not w.isalpha())
# 'Io to the beach with my'
print(sent)
