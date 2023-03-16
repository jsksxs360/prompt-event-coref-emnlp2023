export OUTPUT_DIR=./results/

python3 run_selector.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=longformer \
    --model_checkpoint=../../PT_MODELS/allenai/longformer-large-4096/ \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --pred_test_file=../../data/epoch_3_test_pred_events.json \
    --max_seq_length=4096 \
    --learning_rate=1e-5 \
    --num_train_epochs=30 \
    --batch_size=1 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42