export OUTPUT_DIR=./longformer_m_hta_hn_512_product_cosine_results/

python3 run_mix_prompt.py \
    --output_dir=$OUTPUT_DIR \
    --prompt_type=m_hta_hn \
    --select_arg_strategy=no_filter \
    --matching_style=product_cosine \
    --cosine_space_dim=64 \
    --cosine_slices=128 \
    --cosine_factor=4 \
    --model_type=longformer \
    --model_checkpoint=../../PT_MODELS/allenai/longformer-large-4096/ \
    --train_file=../../data/train_filtered.json \
    --train_file_with_cos=../../data/train_filtered_with_cos.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --train_simi_file=../../data/KnowledgeExtraction/simi_files/simi_omni_train_related_info_0.75.json \
    --dev_simi_file=../../data/KnowledgeExtraction/simi_files/simi_omni_dev_related_info_0.75.json \
    --test_simi_file=../../data/KnowledgeExtraction/simi_files/simi_omni_gold_test_related_info_0.75.json \
    --pred_test_simi_file=../../data/KnowledgeExtraction/simi_files/simi_omni_epoch_3_test_related_info_0.75.json \
    --train_data_type=tiny \
    --sample_strategy=nearmiss \
    --neg_top_k=3 \
    --max_seq_length=512 \
    --learning_rate=1e-5 \
    --num_train_epochs=10 \
    --batch_size=4 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42