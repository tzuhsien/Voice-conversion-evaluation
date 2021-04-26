#! /bin/bash
# $1: GPU device
# $2: Corpus dir
# $3: Path of extracted embeddings
# $4: How many positive samples
# $5: Path of pairs for calculate eer
# $6: Corpus name
# $7: Path of threshold
CUDA_VISIBLE_DEVICES=$1 python resemblyzer_extract_embedding.py $2 -o $3 
CUDA_VISIBLE_DEVICES=$1 python prepare_eer_samples.py $3 -n $4 -o $5
CUDA_VISIBLE_DEVICES=$1 python calculate_eer.py -d $5 -n $6 -o $7