export OUTPUT_DIR=./longformer_cnn_softmax_results/

python3 run_td_softmax.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=longformer \
    --model_checkpoint=../../PT_MODELS/allenai/longformer-large-4096/ \
    --use_addition_layer=cnn \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=4096 \
    --learning_rate=1e-5 \
    --num_train_epochs=15 \
    --batch_size=1 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42