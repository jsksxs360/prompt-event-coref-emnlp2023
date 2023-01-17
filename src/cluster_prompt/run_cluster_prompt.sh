export OUTPUT_DIR=./longformer_pb_d_512_results/

python3 run_cluster_prompt.py \
    --output_dir=$OUTPUT_DIR \
    --prompt_type=sb_d \
    --model_type=longformer \
    --model_checkpoint=../../PT_MODELS/allenai/longformer-large-4096/ \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --train_file_with_cos=../../data/train_filtered_with_cos.json \
    --dev_file_with_cos=../../data/dev_filtered_with_cos.json \
    --test_file_with_cos=../../data/test_filtered_with_cos.json \
    --pos_r=1. \
    --neg_r=1.5 \
    --max_seq_length=512 \
    --learning_rate=1e-5 \
    --num_train_epochs=20 \
    --batch_size=4 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42