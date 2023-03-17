from sklearn.metrics import classification_report
from collections import defaultdict
import sys
sys.path.append('../../')
from src.analysis.utils import get_event_pair_set

gold_coref_file = '../../data/test.json'
pred_coref_file = '../clustering/event-event/epoch_10_dev_f1_73.3789_weights.bin_longformer_m_ht_hc_test_pred_corefs.json'

def all_metrics(gold_coref_file, pred_coref_file):
    gold_coref_results, pred_coref_results = get_event_pair_set(gold_coref_file, pred_coref_file)
    all_event_pairs = [] # (gold_coref, pred_coref)
    for doc_id in gold_coref_results:
        # {e_i_start-e_j_start: (coref, sent_dist, e_i_link_len, e_j_link_len)}
        gold_unrecognized_event_pairs, gold_recognized_event_pairs = (
            gold_coref_results[doc_id]['unrecognized_event_pairs'], 
            gold_coref_results[doc_id]['recognized_event_pairs']
        )
        # {e_i_start-e_j_start: (coref, sent_dist, e_i_link_len, e_j_link_len)}
        pred_recognized_event_pairs, pred_wrong_event_pairs = (
            pred_coref_results[doc_id]['recognized_event_pairs'], 
            pred_coref_results[doc_id]['wrong_event_pairs']
        )
        for pair_results in gold_unrecognized_event_pairs.values():
            all_event_pairs.append([str(pair_results[0]), '2'])
        for pair_id, pair_results in gold_recognized_event_pairs.items():
            all_event_pairs.append([str(pair_results[0]), str(pred_recognized_event_pairs[pair_id][0])])
        for pair_id, pair_results in pred_wrong_event_pairs.items():
            all_event_pairs.append(['0', str(pair_results[0])])
    y_true, y_pred = [res[0] for res in all_event_pairs], [res[1] for res in all_event_pairs]
    metrics = {'ALL': classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']}
    return metrics

print(all_metrics(gold_coref_file, pred_coref_file))