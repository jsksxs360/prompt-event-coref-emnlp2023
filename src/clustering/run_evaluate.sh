export OUTPUT_DIR=./TEMP/

python3 run_evaluate.py \
    --output_dir=$OUTPUT_DIR \
    --test_golden_filepath=../../data/test.json \
    --test_cluster_filepath=final_clusters/longformer_sb_d_512_event_tiny3w-longformer_sb_d_512_event-test_pred_clusters.json \
    --golden_conll_filename=gold_test.conll \
    --pred_conll_filename=pred_test.conll \
    --do_evaluate