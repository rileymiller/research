#!/bin/bash

# topics=(10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
topics=(11)
for i in "${topics[@]}"
do
    # train description model
    echo "INFO: training ETM on descriptions"

    python main.py --mode train --dataset description_embeddings_100 --data_path data/description_embeddings_100/ --emb_path embeddings/wdv_descriptions --num_topics $i --train_embeddings 0 --epochs 1000 --num_words 11

    echo "INFO: finished training ETM on descriptions with pretrained embeddings"

    echo "INFO: evaluating ETM on descriptions with pretrained embeddings"

    python main.py --mode eval --dataset description_embeddings_100 --data_path data/description_embeddings_100/ --num_topics $i --num_words 11 --train_embeddings 1 --tc 1 --load_from results/etm_description_embeddings_100_K_${i}_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_0 | tail -n $i | sed -E 's/Topic \w+: \[//g' | sed -E 's/\]//g' | sed -E s/"'"//g | tee "../results/etm/descriptions/description-etm-${i}topics.txt"

    echo "INFO: finished evaluating ETM descriptions with pretrained embeddings and $i topics"
done
