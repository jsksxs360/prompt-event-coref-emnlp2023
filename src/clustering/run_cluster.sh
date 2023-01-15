
export OUTPUT_DIR=./TEMP/

python3 run_cluster.py \
    --output_dir=$OUTPUT_DIR \
    --test_golden_filepath=../../data/test.json \
    --test_pred_filepath=event-event/longformer_hq_d_512_mask_event_test_pred_corefs.json \
    --golden_conll_filename=gold_test.conll \
    --pred_conll_filename=pred_test.conll \
    --do_evaluate