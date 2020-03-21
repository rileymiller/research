#!/bin/bash

# declare the number of topics to text
array=( 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
for i in "${array[@]}"
do
    echo "INFO: running run_lda2vec.py for descriptions with $i topics"

    # train run_lda2vec and grab the n topics generated and pipe to a file
    python run_lda2vec.py --num_topics $i --data_type description | tail -n $(($i+1)) | head -n $i | sed -E 's/^Topic \w+ : //g' | tee "results/lda2vec/descriptions/description-lda2vec-${i}topics.txt"

    echo "INFO: finished running run_lda2vec.py for descriptions with $i topics"


    echo "INFO: running run_lda2vec.py for titles with $i topics"

    # train run_lda2vec and grab the n topics generated and pipe to a file
    python run_lda2vec.py --num_topics $i --data_type title | tail -n $(($i+1)) | head -n $i | sed -E 's/^Topic \w+ : //g' | tee "results/lda2vec/titles/title-lda2vec-${i}topics.txt"

    echo "INFO: finished running run_lda2vec.py for titles with $i topics"
done
