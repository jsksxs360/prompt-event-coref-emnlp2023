from sklearn.metrics import classification_report
import sys
sys.path.append('../../')
from src.analysis.utils import get_event_pair_set

def all_metrics(
    gold_coref_file, gold_simi_coref_file, 
    pred_coref_file, pred_simi_coref_file
    ):
    _, gold_coref_results, pred_coref_results = get_event_pair_set(
        gold_coref_file, gold_simi_coref_file, 
        pred_coref_file, pred_simi_coref_file, 
        mode='easy'
    )
    all_event_pairs = [] # (gold_coref, pred_coref)
    for doc_id in gold_coref_results:
        gold_unrecognized_event_pairs, gold_recognized_event_pairs = (
            gold_coref_results[doc_id]['unrecognized_event_pairs'], 
            gold_coref_results[doc_id]['recognized_event_pairs']
        )
        pred_recognized_event_pairs, pred_wrong_event_pairs = (
            pred_coref_results[doc_id]['recognized_event_pairs'], 
            pred_coref_results[doc_id]['wrong_event_pairs']
        )
        for pair_results in gold_unrecognized_event_pairs.values():
            all_event_pairs.append([str(pair_results['coref']), '2'])
        for pair_id, pair_results in gold_recognized_event_pairs.items():
            all_event_pairs.append([str(pair_results['coref']), str(pred_recognized_event_pairs[pair_id]['coref'])])
        for pair_id, pair_results in pred_wrong_event_pairs.items():
            all_event_pairs.append(['0', str(pair_results['coref'])])
    y_true, y_pred = [res[0] for res in all_event_pairs], [res[1] for res in all_event_pairs]
    metrics = {'ALL': classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']}
    return metrics

# gold_coref_file = '../../data/test.json'
# gold_simi_coref_file = '../../data/KnowledgeExtraction/simi_files/simi_omni_gold_test_related_info_0.75.json'
# pred_coref_file = '../clustering/event-event/epoch_2_dev_f1_69.5294_weights.bin_roberta_ma_remove-subtype-match_test_pred_corefs.json'
# pred_simi_coref_file = '../../data/KnowledgeExtraction/simi_files/simi_omni_epoch_3_test_related_info_0.75.json'
# print(all_metrics(gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file))

def different_arg_status_metrics_easy(
    gold_coref_file, gold_simi_coref_file, 
    pred_coref_file, pred_simi_coref_file
    ):
    event_pair_info, gold_coref_results, pred_coref_results = get_event_pair_set(
        gold_coref_file, gold_simi_coref_file, 
        pred_coref_file, pred_simi_coref_file, 
        mode='easy'
    )
    no, one_arg, both_arg, arg = [], [], [], []
    unbalance = []
    for doc_id in gold_coref_results:
        gold_unrecognized_event_pairs, gold_recognized_event_pairs = (
            gold_coref_results[doc_id]['unrecognized_event_pairs'], 
            gold_coref_results[doc_id]['recognized_event_pairs']
        )
        pred_recognized_event_pairs, pred_wrong_event_pairs = (
            pred_coref_results[doc_id]['recognized_event_pairs'], 
            pred_coref_results[doc_id]['wrong_event_pairs']
        )
        for pair_id, pair_results in gold_unrecognized_event_pairs.items():
            pair_arg_info = event_pair_info[doc_id][pair_id]
            e_i_has_arg = pair_arg_info['e_i_has_part'] or pair_arg_info['e_i_has_place']
            e_j_has_arg = pair_arg_info['e_j_has_part'] or pair_arg_info['e_j_has_place']
            pair_coref = [str(pair_results['coref']), '2']
            if not e_i_has_arg and not e_j_has_arg:
                no.append(pair_coref)
            else:
                if (pair_arg_info['e_i_has_part'] != pair_arg_info['e_j_has_part']) and \
                   (pair_arg_info['e_i_has_place'] != pair_arg_info['e_j_has_place']):
                    unbalance.append(pair_coref)
                arg.append(pair_coref)
                if e_i_has_arg and e_j_has_arg:
                    both_arg.append(pair_coref)
                else:
                    one_arg.append(pair_coref)
        for pair_id, pair_results in gold_recognized_event_pairs.items():
            pair_arg_info = event_pair_info[doc_id][pair_id]
            e_i_has_arg = pair_arg_info['e_i_has_part'] or pair_arg_info['e_i_has_place']
            e_j_has_arg = pair_arg_info['e_j_has_part'] or pair_arg_info['e_j_has_place']
            pair_coref = [str(pair_results['coref']), str(pred_recognized_event_pairs[pair_id]['coref'])]
            if not e_i_has_arg and not e_j_has_arg:
                no.append(pair_coref)
            else:
                if (pair_arg_info['e_i_has_part'] != pair_arg_info['e_j_has_part']) and \
                   (pair_arg_info['e_i_has_place'] != pair_arg_info['e_j_has_place']):
                    unbalance.append(pair_coref)
                arg.append(pair_coref)
                if e_i_has_arg and e_j_has_arg:
                    both_arg.append(pair_coref)
                else:
                    one_arg.append(pair_coref)
        for pair_id, pair_results in pred_wrong_event_pairs.items():
            pair_arg_info = event_pair_info[doc_id][pair_id]
            e_i_has_arg = pair_arg_info['e_i_has_part'] or pair_arg_info['e_i_has_place']
            e_j_has_arg = pair_arg_info['e_j_has_part'] or pair_arg_info['e_j_has_place']
            pair_coref = ['0', str(pair_results['coref'])]
            if not e_i_has_arg and not e_j_has_arg:
                no.append(pair_coref)
            else:
                if (pair_arg_info['e_i_has_part'] != pair_arg_info['e_j_has_part']) and \
                   (pair_arg_info['e_i_has_place'] != pair_arg_info['e_j_has_place']):
                    unbalance.append(pair_coref)
                arg.append(pair_coref)
                if e_i_has_arg and e_j_has_arg:
                    both_arg.append(pair_coref)
                else:
                    one_arg.append(pair_coref)
    metrics = {}
    y_true, y_pred = [res[0] for res in no], [res[1] for res in no]
    metrics['no_arg'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in one_arg], [res[1] for res in one_arg]
    metrics['one_arg'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in both_arg], [res[1] for res in both_arg]
    metrics['both_arg'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in arg], [res[1] for res in arg]
    metrics['arg'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in unbalance], [res[1] for res in unbalance]
    metrics['unbal'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    return metrics

# gold_coref_file = '../../data/test.json'
# gold_simi_coref_file = '../../data/KnowledgeExtraction/simi_files/simi_omni_gold_test_related_info_0.75.json'
# pred_coref_file = '../clustering/event-event/epoch_2_dev_f1_71.5876_weights.bin_roberta_m_hta_hn_test_pred_corefs.json'
# pred_simi_coref_file = '../../data/KnowledgeExtraction/simi_files/simi_omni_epoch_3_test_related_info_0.75.json'
# print(different_arg_status_metrics_easy(gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file))

def error_analysis(
    gold_coref_file, gold_simi_coref_file, 
    pred_coref_file, pred_simi_coref_file
    ):
    _, gold_coref_results, pred_coref_results = get_event_pair_set(
        gold_coref_file, gold_simi_coref_file, 
        pred_coref_file, pred_simi_coref_file, 
        mode='easy'
    )
    total_num, total_correct_num = 0, 0
    unrecognized_num, misidentified_num, misjudged_num = 0, 0, 0
    coref2nocoref_num, nocoref2coref_num = 0, 0
    for doc_id in gold_coref_results:
        gold_unrecognized_event_pairs, gold_recognized_event_pairs = (
            gold_coref_results[doc_id]['unrecognized_event_pairs'], 
            gold_coref_results[doc_id]['recognized_event_pairs']
        )
        pred_recognized_event_pairs, pred_wrong_event_pairs = (
            pred_coref_results[doc_id]['recognized_event_pairs'], 
            pred_coref_results[doc_id]['wrong_event_pairs']
        )
        unrecognized, misidentified, recognized = (
            len(gold_unrecognized_event_pairs), 
            len(pred_wrong_event_pairs), 
            len(gold_recognized_event_pairs)
        )
        total_num += (unrecognized + misidentified + recognized)
        unrecognized_num += unrecognized
        misidentified_num += misidentified
        for pair_id, pair_results in gold_recognized_event_pairs.items():
            true_label, pred_label = str(pair_results['coref']), str(pred_recognized_event_pairs[pair_id]['coref'])
            if true_label == pred_label:
                total_correct_num += 1
            else:
                misjudged_num += 1
                if true_label == '1':
                    coref2nocoref_num += 1
                else:
                    nocoref2coref_num += 1
    total_wrong_num = unrecognized_num + misidentified_num + misjudged_num
    assert total_correct_num + total_wrong_num == total_num
    print(f'ACC: {total_correct_num} / {total_num} = {(total_correct_num/total_num)*100:.2f}%, total error: {total_wrong_num}')
    print('errors ====>')
    print(f'unrecognized: {unrecognized_num} / {total_wrong_num} = {(unrecognized_num/total_wrong_num)*100:.2f}%')
    print(f'misidentified: {misidentified_num} / {total_wrong_num} = {(misidentified_num/total_wrong_num)*100:.2f}%')
    print(f'recognize error: {unrecognized_num + misidentified_num} / {total_wrong_num} = {((unrecognized_num + misidentified_num)/total_wrong_num)*100:.2f}%')
    print(f'misjudged: {misjudged_num} / {total_wrong_num} = {(misjudged_num/total_wrong_num)*100:.2f}%')
    print(
        f'coref2nocoref: {coref2nocoref_num} / {misjudged_num} = {(coref2nocoref_num/misjudged_num)*100:.2f}%, '
        f'nocoref2coref: {nocoref2coref_num} / {misjudged_num} = {(nocoref2coref_num/misjudged_num)*100:.2f}%'
    )

gold_coref_file = '../../data/test.json'
gold_simi_coref_file = '../../data/KnowledgeExtraction/simi_files/simi_omni_gold_test_related_info_0.75.json'
pred_coref_file = '../clustering/event-event/epoch_2_dev_f1_71.5876_weights.bin_roberta_m_hta_hn_test_pred_corefs.json'
pred_simi_coref_file = '../../data/KnowledgeExtraction/simi_files/simi_omni_epoch_3_test_related_info_0.75.json'
error_analysis(gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file)