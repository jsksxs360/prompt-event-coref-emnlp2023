export OUTPUT_DIR=./longformer_sb_d_512_event_results/

python3 run_prompt.py \
    --output_dir=$OUTPUT_DIR \
    --prompt_type=sb_d \
    --model_type=longformer \
    --longformer_global_att=event \
    --model_checkpoint=../../PT_MODELS/allenai/longformer-large-4096/ \
    --train_file=../../data/train_filtered.json \
    --train_file_with_cos=../../data/train_filtered_with_cos.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --train_data_type=tiny \
    --neg_top_k=10 \
    --max_seq_length=512 \
    --learning_rate=1e-5 \
    --num_train_epochs=20 \
    --batch_size=4 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42