# Generate pre-fitted skipgram embeddings
```
python skipgram.py --data_file ../datasets/parsed_full_titles.txt --emb_file embeddings/wdv_title
```
(generate for both title and description)

# Train the model with the prefitted embeddings

### Description Training
```
python main.py --mode train --dataset description_embeddings_100 --data_path data/description_embeddings_100/ --emb_path embeddings/wdv_descriptions --num_topics 25 --train_embeddings 0 --epochs 1000 --num_words 11
```

### Title Training
```
python main.py --mode train --dataset title_embeddings_100 --data_path data/title_embeddings_100/ --emb_path embeddings/wdv_titles --num_topics 25 --train_embeddings 0 --epochs 1000 --num_words 11
```
# Evaluate the model trained on prefitted embeddings

### Description Evaluation
```
python main.py --mode eval --dataset description_embeddings_100 --data_path data/description_embeddings_100/ --num_topics 25 --num_words 11 --train_embeddings 1 --tc 1 --load_from results/etm_description_embeddings_100_K_25_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_0
```

### Title Evaluation
```
python main.py --mode eval --dataset title_embeddings_100 --data_path data/title_embeddings_100/ --num_topics 25 --num_words 11 --train_embeddings 1 --tc 1 --load_from results/etm_title_embeddings_100_K_25_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_0
```

