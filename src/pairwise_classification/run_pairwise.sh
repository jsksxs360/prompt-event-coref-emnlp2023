export OUTPUT_DIR=./bert_product_cosine_results/

python3 run_pairwise.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_subtype=normal_model \
    --model_checkpoint=../../PT_MODELS/bert-large-cased/ \
    --matching_style=event \
    --train_file=../../data/train_filtered.json \
    --train_file_with_cos=../../data/train_filtered_with_cos.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --train_data_type=tiny \
    --neg_top_k=10 \
    --max_seq_length=512 \
    --learning_rate=1e-5 \
    --num_train_epochs=10 \
    --batch_size=4 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42