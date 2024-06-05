#!/bin/bash

# datasets=("CanParl" "Flights" "mooc" "SocialEvo" "UNtrade" "USLegis" "Contacts" "enron" "lastfm" "reddit" "uci" "UNvote" "wikipedia")
# datasets=( "wikipedia" "reddit" "uci" )
# datasets=("enron" "mooc" "Contacts")
datasets=("uci")
S=10  

for dataset in "${datasets[@]}" 
do
    echo "Running TGAT on dataset: $dataset"
    # Real data
    for sample in {1..5}; do
        python -u learn_edge.py -d "$dataset" --bs 200 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world
    done

    # distorted data: all samples
    for sample in {1..5}; do
        distort="intense_5_${sample}_"
        python -u learn_edge.py -d "$dataset" --distortion "$distort" --bs 200 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world
    done

    for sample in {1..5}; do
        distort="shuffle_${sample}_"
        python -u learn_edge.py -d "$dataset" --distortion "$distort" --bs 200 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world
    done
done


