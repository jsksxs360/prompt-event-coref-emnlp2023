from sklearn.metrics import classification_report
import sys
sys.path.append('../../')
from src.analysis.utils import get_event_pair_set

def all_metrics(
    prompt_type, select_arg_strategy, 
    gold_coref_file, gold_simi_coref_file, 
    pred_coref_file, pred_simi_coref_file
    ):
    _, gold_coref_results, pred_coref_results = get_event_pair_set(
        prompt_type, select_arg_strategy, 
        gold_coref_file, gold_simi_coref_file, 
        pred_coref_file, pred_simi_coref_file
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

# prompt_type = 'm_hta_hn'
# select_arg_strategy = 'no_filter'
# gold_coref_file = '../../data/test.json'
# gold_simi_coref_file = '../../data/KnowledgeExtraction/simi_gold_test_related_info_0.75.json'
# pred_coref_file = '../clustering/event-event/epoch_5_dev_f1_73.5794_weights.bin_longformer_m_hta_hn_test_pred_corefs.json'
# pred_simi_coref_file = '../../data/KnowledgeExtraction/simi_epoch_3_test_related_info_0.75.json'
# print(all_metrics(prompt_type, select_arg_strategy, gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file))

def different_has_arg_status_metrics(prompt_type, select_arg_strategy, gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file):
    event_pair_info, gold_coref_results, pred_coref_results = get_event_pair_set(
        prompt_type, select_arg_strategy, 
        gold_coref_file, gold_simi_coref_file, 
        pred_coref_file, pred_simi_coref_file
    )
    both_no, one_has, both_have = [], [], []
    one_has_part, one_has_place, one_has_part_place = [], [], []
    both_have_part, both_have_place, both_have_part_place, both_have_unbalance = [], [], [], []
    both_have_unbalance_have_part, both_have_unbalance_have_place, both_have_unbalance_other = [], [], []
    for doc_id, gold_coref_result_dict in gold_coref_results.items():
        gold_unrecognized_event_pairs, gold_recognized_event_pairs = (
            gold_coref_result_dict['unrecognized_event_pairs'], 
            gold_coref_result_dict['recognized_event_pairs']
        )
        pred_coref_result_dict = pred_coref_results[doc_id]
        pred_recognized_event_pairs, pred_wrong_event_pairs = (
            pred_coref_result_dict['recognized_event_pairs'], 
            pred_coref_result_dict['wrong_event_pairs']
        )
        for pair_id, pair_coref_info in gold_unrecognized_event_pairs.items():
            pair_results = event_pair_info[doc_id][pair_id]
            e_i_has_arg = pair_results['e_i_has_part'] or pair_results['e_i_has_place']
            e_j_has_arg = pair_results['e_j_has_part'] or pair_results['e_j_has_place']
            pair_coref = [str(pair_coref_info['coref']), '2']
            if not e_i_has_arg and not e_j_has_arg: # both have no argument
                both_no.append(pair_coref)
            else:
                if e_i_has_arg and e_j_has_arg: # both have arguments
                    both_have.append(pair_coref)
                    if pair_results['e_i_has_part'] and pair_results['e_i_has_place'] and pair_results['e_j_has_part'] and pair_results['e_j_has_place']:
                        both_have_part_place.append(pair_coref)
                    elif (pair_results['e_i_has_part'] and pair_results['e_j_has_part']) and (not pair_results['e_i_has_place'] and not pair_results['e_j_has_place']):
                        both_have_part.append(pair_coref)
                    elif (pair_results['e_i_has_place'] and pair_results['e_j_has_place']) and (not pair_results['e_i_has_part'] and not pair_results['e_j_has_part']):
                        both_have_place.append(pair_coref)
                    else:
                        both_have_unbalance.append(pair_coref)
                        if pair_results['e_i_has_part'] and pair_results['e_j_has_part']:
                            both_have_unbalance_have_part.append(pair_coref)
                        elif pair_results['e_i_has_place'] and pair_results['e_j_has_place']:
                            both_have_unbalance_have_place.append(pair_coref)
                        else:
                            both_have_unbalance_other.append(pair_coref)
                else: # one has argument
                    one_has.append(pair_coref)
                    if e_i_has_arg:
                        if pair_results['e_i_has_part'] and pair_results['e_i_has_place']:
                            one_has_part_place.append(pair_coref)
                        elif pair_results['e_i_has_part']:
                            one_has_part.append(pair_coref)
                        else:
                            one_has_place.append(pair_coref)
                    else:
                        if pair_results['e_j_has_part'] and pair_results['e_j_has_place']:
                            one_has_part_place.append(pair_coref)
                        elif pair_results['e_j_has_part']:
                            one_has_part.append(pair_coref)
                        else:
                            one_has_place.append(pair_coref)
        for pair_id, pair_coref_info in pred_recognized_event_pairs.items():
            pair_results = event_pair_info[doc_id][pair_id]
            e_i_has_arg = pair_results['e_i_has_part'] or pair_results['e_i_has_place']
            e_j_has_arg = pair_results['e_j_has_part'] or pair_results['e_j_has_place']
            pair_coref = [str(gold_recognized_event_pairs[pair_id]['coref']), str(pair_coref_info['coref'])]
            if not e_i_has_arg and not e_j_has_arg: # both have no argument
                both_no.append(pair_coref)
            else:
                if e_i_has_arg and e_j_has_arg: # both have arguments
                    both_have.append(pair_coref)
                    if pair_results['e_i_has_part'] and pair_results['e_i_has_place'] and pair_results['e_j_has_part'] and pair_results['e_j_has_place']:
                        both_have_part_place.append(pair_coref)
                    elif (pair_results['e_i_has_part'] and pair_results['e_j_has_part']) and (not pair_results['e_i_has_place'] and not pair_results['e_j_has_place']):
                        both_have_part.append(pair_coref)
                    elif (pair_results['e_i_has_place'] and pair_results['e_j_has_place']) and (not pair_results['e_i_has_part'] and not pair_results['e_j_has_part']):
                        both_have_place.append(pair_coref)
                    else:
                        both_have_unbalance.append(pair_coref)
                        if pair_results['e_i_has_part'] and pair_results['e_j_has_part']:
                            both_have_unbalance_have_part.append(pair_coref)
                        elif pair_results['e_i_has_place'] and pair_results['e_j_has_place']:
                            both_have_unbalance_have_place.append(pair_coref)
                        else:
                            both_have_unbalance_other.append(pair_coref)
                else: # one has argument
                    one_has.append(pair_coref)
                    if e_i_has_arg:
                        if pair_results['e_i_has_part'] and pair_results['e_i_has_place']:
                            one_has_part_place.append(pair_coref)
                        elif pair_results['e_i_has_part']:
                            one_has_part.append(pair_coref)
                        else:
                            one_has_place.append(pair_coref)
                    else:
                        if pair_results['e_j_has_part'] and pair_results['e_j_has_place']:
                            one_has_part_place.append(pair_coref)
                        elif pair_results['e_j_has_part']:
                            one_has_part.append(pair_coref)
                        else:
                            one_has_place.append(pair_coref)
        for pair_id, pair_coref_info in pred_wrong_event_pairs.items():
            pair_results = event_pair_info[doc_id][pair_id]
            e_i_has_arg = pair_results['e_i_has_part'] or pair_results['e_i_has_place']
            e_j_has_arg = pair_results['e_j_has_part'] or pair_results['e_j_has_place']
            pair_coref = ['0', str(pair_coref_info['coref'])]
            if not e_i_has_arg and not e_j_has_arg: # both have no argument
                both_no.append(pair_coref)
            else:
                if e_i_has_arg and e_j_has_arg: # both have arguments
                    both_have.append(pair_coref)
                    if pair_results['e_i_has_part'] and pair_results['e_i_has_place'] and pair_results['e_j_has_part'] and pair_results['e_j_has_place']:
                        both_have_part_place.append(pair_coref)
                    elif (pair_results['e_i_has_part'] and pair_results['e_j_has_part']) and (not pair_results['e_i_has_place'] and not pair_results['e_j_has_place']):
                        both_have_part.append(pair_coref)
                    elif (pair_results['e_i_has_place'] and pair_results['e_j_has_place']) and (not pair_results['e_i_has_part'] and not pair_results['e_j_has_part']):
                        both_have_place.append(pair_coref)
                    else:
                        both_have_unbalance.append(pair_coref)
                        if pair_results['e_i_has_part'] and pair_results['e_j_has_part']:
                            both_have_unbalance_have_part.append(pair_coref)
                        elif pair_results['e_i_has_place'] and pair_results['e_j_has_place']:
                            both_have_unbalance_have_place.append(pair_coref)
                        else:
                            both_have_unbalance_other.append(pair_coref)
                else: # one has argument
                    one_has.append(pair_coref)
                    if e_i_has_arg:
                        if pair_results['e_i_has_part'] and pair_results['e_i_has_place']:
                            one_has_part_place.append(pair_coref)
                        elif pair_results['e_i_has_part']:
                            one_has_part.append(pair_coref)
                        else:
                            one_has_place.append(pair_coref)
                    else:
                        if pair_results['e_j_has_part'] and pair_results['e_j_has_place']:
                            one_has_part_place.append(pair_coref)
                        elif pair_results['e_j_has_part']:
                            one_has_part.append(pair_coref)
                        else:
                            one_has_place.append(pair_coref)
    metrics = {}
    y_true, y_pred = [res[0] for res in both_no], [res[1] for res in both_no]
    metrics['both_no'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in one_has], [res[1] for res in one_has]
    metrics['one_has'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in both_have], [res[1] for res in both_have]
    metrics['both_have'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in one_has_part], [res[1] for res in one_has_part]
    metrics['one_has_part'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in one_has_place], [res[1] for res in one_has_place]
    metrics['one_has_place'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in one_has_part_place], [res[1] for res in one_has_part_place]
    metrics['one_has_part_place'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in both_have_part], [res[1] for res in both_have_part]
    metrics['both_have_part'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in both_have_place], [res[1] for res in both_have_place]
    metrics['both_have_place'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in both_have_part_place], [res[1] for res in both_have_part_place]
    metrics['both_have_part_place'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in both_have_unbalance], [res[1] for res in both_have_unbalance]
    metrics['both_have_unbalance'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in both_have_unbalance_have_part], [res[1] for res in both_have_unbalance_have_part]
    metrics['both_have_unbalance_have_part'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in both_have_unbalance_have_place], [res[1] for res in both_have_unbalance_have_place]
    metrics['both_have_unbalance_have_place'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in both_have_unbalance_other], [res[1] for res in both_have_unbalance_other]
    metrics['both_have_unbalance_other'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    return metrics

# prompt_type = 'm_hta_hn'
# select_arg_strategy = 'no_filter'
# gold_coref_file = '../../data/test.json'
# gold_simi_coref_file = '../../data/KnowledgeExtraction/simi_gold_test_related_info_0.75.json'
# pred_coref_file = '../clustering/event-event/epoch_5_dev_f1_73.5794_weights.bin_longformer_m_hta_hn_test_pred_corefs.json'
# pred_simi_coref_file = '../../data/KnowledgeExtraction/simi_epoch_3_test_related_info_0.75.json'
# print(different_has_arg_status_metrics(prompt_type, select_arg_strategy, gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file))

# def compare_two_results(prompt_type, select_arg_strategy, gold_coref_file, gold_simi_coref_file, pred_coref_file_1, pred_coref_file_2, pred_simi_coref_file):
#     gold_coref_results_1, pred_coref_results_1 = get_event_pair_set(prompt_type, select_arg_strategy, gold_coref_file, gold_simi_coref_file, pred_coref_file_1, pred_simi_coref_file)
#     gold_coref_results_2, pred_coref_results_2 = get_event_pair_set(prompt_type, select_arg_strategy, gold_coref_file, gold_simi_coref_file, pred_coref_file_2, pred_simi_coref_file)
#     c2w = 
    
#     for doc_id in gold_coref_results_1:
#         gold_recognized_event_pairs_1 = gold_coref_results_1[doc_id]['recognized_event_pairs']
#         pred_coref_result_dict_1 = pred_coref_results_1[doc_id]
#         pred_recognized_event_pairs_1, pred_wrong_event_pairs_1 = (
#             pred_coref_result_dict_1['recognized_event_pairs'], 
#             pred_coref_result_dict_1['wrong_event_pairs']
#         )
#         gold_recognized_event_pairs_2 = gold_coref_results_2[doc_id]['recognized_event_pairs']
#         pred_coref_result_dict_2 = pred_coref_results_2[doc_id]
#         pred_recognized_event_pairs_2, pred_wrong_event_pairs_2 = (
#             pred_coref_result_dict_2['recognized_event_pairs'], 
#             pred_coref_result_dict_2['wrong_event_pairs']
#         )
#         for pair_id in pred_recognized_event_pairs_1:
#             assert gold_recognized_event_pairs_1[pair_id]['coref'] == gold_recognized_event_pairs_2[pair_id]['coref']
#             pair_coref_1 = [str(gold_recognized_event_pairs_1[pair_id]['coref']), str(pred_recognized_event_pairs_1[pair_id]['coref'])]
#             pair_coref_bool_1 = pair_coref_1[0] == pair_coref_1[1]
#             pair_coref_2 = [str(gold_recognized_event_pairs_2[pair_id]['coref']), str(pred_recognized_event_pairs_2[pair_id]['coref'])]
#             pair_coref_bool_2 = pair_coref_2[0] == pair_coref_2[1]
#             info = {
#                 'e_i': e_i_pretty_sent, 'e_j_pretty_sent': e_j_pretty_sent, 
#             }

            
#         for pair_id, pair_results in pred_wrong_event_pairs.items():
#             e_i_has_arg = pair_results['e_i_has_part'] or pair_results['e_i_has_place']
#             e_j_has_arg = pair_results['e_j_has_part'] or pair_results['e_j_has_place']
#             pair_coref = ['0', str(pair_results['coref'])]
#             if not e_i_has_arg and not e_j_has_arg: # both have no argument
#                 both_no.append(pair_coref)
#             else:
#                 if e_i_has_arg and e_j_has_arg: # both have arguments
#                     both_have.append(pair_coref)
#                     if pair_results['e_i_has_part'] and pair_results['e_i_has_place'] and pair_results['e_j_has_part'] and pair_results['e_j_has_place']:
#                         both_have_part_place.append(pair_coref)
#                     elif (pair_results['e_i_has_part'] and pair_results['e_j_has_part']) and (not pair_results['e_i_has_place'] and not pair_results['e_j_has_place']):
#                         both_have_part.append(pair_coref)
#                     elif (pair_results['e_i_has_place'] and pair_results['e_j_has_place']) and (not pair_results['e_i_has_part'] and not pair_results['e_j_has_part']):
#                         both_have_place.append(pair_coref)
#                     else:
#                         both_have_unbalance.append(pair_coref)
#                         if pair_results['e_i_has_part'] and pair_results['e_j_has_part']:
#                             both_have_unbalance_have_part.append(pair_coref)
#                         elif pair_results['e_i_has_place'] and pair_results['e_j_has_place']:
#                             both_have_unbalance_have_place.append(pair_coref)
#                         else:
#                             both_have_unbalance_other.append(pair_coref)
#                 else: # one has argument
#                     one_has.append(pair_coref)
#                     if e_i_has_arg:
#                         if pair_results['e_i_has_part'] and pair_results['e_i_has_place']:
#                             one_has_part_place.append(pair_coref)
#                         elif pair_results['e_i_has_part']:
#                             one_has_part.append(pair_coref)
#                         else:
#                             one_has_place.append(pair_coref)
#                     else:
#                         if pair_results['e_j_has_part'] and pair_results['e_j_has_place']:
#                             one_has_part_place.append(pair_coref)
#                         elif pair_results['e_j_has_part']:
#                             one_has_part.append(pair_coref)
#                         else:
#                             one_has_place.append(pair_coref)

























































def different_find_arg_status_metrics(gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file):
    gold_coref_results, pred_coref_results = get_event_pair_set(gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file)
    nn2yn, nn2yy, yn2yy, yy = [], [], [], []
    for doc_id, gold_coref_result_dict in gold_coref_results.items():
        gold_unrecognized_event_pairs, gold_recognized_event_pairs = (
            gold_coref_result_dict['unrecognized_event_pairs'], 
            gold_coref_result_dict['recognized_event_pairs']
        )
        pred_coref_result_dict = pred_coref_results[doc_id]
        pred_recognized_event_pairs, pred_wrong_event_pairs = (
            pred_coref_result_dict['recognized_event_pairs'], 
            pred_coref_result_dict['wrong_event_pairs']
        )
        for pair_results in gold_unrecognized_event_pairs.values():
            e_i_has_arg, e_j_has_arg = pair_results['e_i_has_arg'], pair_results['e_j_has_arg']
            e_i_find_arg, e_j_find_arg = pair_results['e_i_find_arg'], pair_results['e_j_find_arg']
            pair_coref = [str(pair_results['coref']), '2']
            if not e_i_has_arg and not e_j_has_arg: # both have no argument
                if e_i_find_arg and e_j_find_arg: # NN->YY
                    nn2yy.append(pair_coref)
                elif e_i_find_arg or e_j_find_arg: # NN->YN
                    nn2yn.append(pair_coref)
            elif e_i_has_arg and e_j_has_arg: # both have arguments
                yy.append(pair_coref)
            else: # one has argument
                if e_i_find_arg or e_j_find_arg: # YN->YY
                    yn2yy.append(pair_coref)
        for pair_id, pair_results in pred_recognized_event_pairs.items():
            e_i_has_arg, e_j_has_arg = pair_results['e_i_has_arg'], pair_results['e_j_has_arg']
            e_i_find_arg, e_j_find_arg = pair_results['e_i_find_arg'], pair_results['e_j_find_arg']
            pair_coref = [str(gold_recognized_event_pairs[pair_id]['coref']), str(pair_results['coref'])]
            if not e_i_has_arg and not e_j_has_arg: # both have no argument
                if e_i_find_arg and e_j_find_arg: # NN->YY
                    nn2yy.append(pair_coref)
                elif e_i_find_arg or e_j_find_arg: # NN->YN
                    nn2yn.append(pair_coref)
            elif e_i_has_arg and e_j_has_arg: # both have arguments
                yy.append(pair_coref)
            else: # one has argument
                if e_i_find_arg or e_j_find_arg: # YN->YY
                    yn2yy.append(pair_coref)
        for pair_id, pair_results in pred_wrong_event_pairs.items():
            e_i_has_arg, e_j_has_arg = pair_results['e_i_has_arg'], pair_results['e_j_has_arg']
            e_i_find_arg, e_j_find_arg = pair_results['e_i_find_arg'], pair_results['e_j_find_arg']
            pair_coref = ['0', str(pair_results['coref'])]
            if not e_i_has_arg and not e_j_has_arg: # both have no argument
                if e_i_find_arg and e_j_find_arg: # NN->YY
                    nn2yy.append(pair_coref)
                elif e_i_find_arg or e_j_find_arg: # NN->YN
                    nn2yn.append(pair_coref)
            elif e_i_has_arg and e_j_has_arg: # both have arguments
                yy.append(pair_coref)
            else: # one has argument
                if e_i_find_arg or e_j_find_arg: # YN->YY
                    yn2yy.append(pair_coref)
    metrics = {}
    y_true, y_pred = [res[0] for res in nn2yn], [res[1] for res in nn2yn]
    metrics['NN->YN'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in nn2yy], [res[1] for res in nn2yy]
    metrics['NN->YY'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in yn2yy], [res[1] for res in yn2yy]
    metrics['YN->YY'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in yy], [res[1] for res in yy]
    metrics['YY'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    return metrics

# gold_coref_file = '../../data/test.json'
# gold_simi_coref_file = '../../data/KnowledgeExtraction/simi_gold_test_related_info_0.75.json'
# pred_coref_file = '../clustering/event-event/epoch_3_dev_f1_72.0731_weights.bin_longformer_m_htao_hn_test_pred_corefs.json'
# pred_simi_coref_file = '../../data/KnowledgeExtraction/simi_epoch_3_test_related_info_0.75.json'
# print(different_find_arg_status_metrics(gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file))

def different_info_balance_metrics(gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file):
    gold_coref_results, pred_coref_results = get_event_pair_set(gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file)
    ba2no, ba2ba, no2ba, no2no = [], [], [], []
    ba2no_nn, ba2no_yn, ba2no_ny, ba2no_yy = [], [], [], []
    ba2ba_nn, ba2ba_yn, ba2ba_ny, ba2ba_yy = [], [], [], []
    for doc_id, gold_coref_result_dict in gold_coref_results.items():
        gold_unrecognized_event_pairs, gold_recognized_event_pairs = (
            gold_coref_result_dict['unrecognized_event_pairs'], 
            gold_coref_result_dict['recognized_event_pairs']
        )
        pred_coref_result_dict = pred_coref_results[doc_id]
        pred_recognized_event_pairs, pred_wrong_event_pairs = (
            pred_coref_result_dict['recognized_event_pairs'], 
            pred_coref_result_dict['wrong_event_pairs']
        )
        for pair_results in gold_unrecognized_event_pairs.values():
            pair_coref = [str(pair_results['coref']), '2']
            e_i_has_part, e_i_has_place = pair_results['e_i_has_part'], pair_results['e_i_has_place']
            e_j_has_part, e_j_has_place = pair_results['e_j_has_part'], pair_results['e_j_has_place']
            e_i_related_part, e_i_related_place = pair_results['e_i_related_part'], pair_results['e_i_related_place']
            e_j_related_part, e_j_related_place = pair_results['e_j_related_part'], pair_results['e_j_related_place']
            e_i_new_part, e_j_new_part = e_i_has_part or e_i_related_part, e_j_has_part or e_j_related_part
            e_i_new_place, e_j_new_place = e_i_has_place or e_i_related_place, e_j_has_place or e_j_related_place
            if e_i_has_part == e_j_has_part and e_i_has_place == e_j_has_place: # balance
                if e_i_new_part == e_j_new_part and e_i_new_place == e_j_new_place: # balance
                    ba2ba.append(pair_coref) # balance -> balance
                    if not e_i_has_part and not e_i_has_place: # NN
                        ba2ba_nn.append(pair_coref)
                    elif e_i_has_part and not e_i_has_place: # YN
                        ba2ba_yn.append(pair_coref)
                    elif not e_i_has_part and e_i_has_place: # NY
                        ba2ba_ny.append(pair_coref)
                    else: # YY
                        ba2ba_yy.append(pair_coref)
                else:
                    ba2no.append(pair_coref) # balance -> no balance
                    if not e_i_has_part and not e_i_has_place: # NN
                        ba2no_nn.append(pair_coref)
                    elif e_i_has_part and not e_i_has_place: # YN
                        ba2no_yn.append(pair_coref)
                    elif not e_i_has_part and e_i_has_place: # NY
                        ba2no_ny.append(pair_coref)
                    else: # YY
                        ba2no_yy.append(pair_coref)
            else: # no balance
                if e_i_new_part == e_j_new_part and e_i_new_place == e_j_new_place: # balance
                    no2ba.append(pair_coref) # no balance -> balance
                else:
                    no2no.append(pair_coref) # no balance -> no balance
        for pair_id, pair_results in pred_recognized_event_pairs.items():
            pair_coref = [str(gold_recognized_event_pairs[pair_id]['coref']), str(pair_results['coref'])]
            e_i_has_part, e_i_has_place = pair_results['e_i_has_part'], pair_results['e_i_has_place']
            e_j_has_part, e_j_has_place = pair_results['e_j_has_part'], pair_results['e_j_has_place']
            e_i_related_part, e_i_related_place = pair_results['e_i_related_part'], pair_results['e_i_related_place']
            e_j_related_part, e_j_related_place = pair_results['e_j_related_part'], pair_results['e_j_related_place']
            e_i_new_part, e_j_new_part = e_i_has_part or e_i_related_part, e_j_has_part or e_j_related_part
            e_i_new_place, e_j_new_place = e_i_has_place or e_i_related_place, e_j_has_place or e_j_related_place
            if e_i_has_part == e_j_has_part and e_i_has_place == e_j_has_place: # balance
                if e_i_new_part == e_j_new_part and e_i_new_place == e_j_new_place: # balance
                    ba2ba.append(pair_coref) # balance -> balance
                    if not e_i_has_part and not e_i_has_place: # NN
                        ba2ba_nn.append(pair_coref)
                    elif e_i_has_part and not e_i_has_place: # YN
                        ba2ba_yn.append(pair_coref)
                    elif not e_i_has_part and e_i_has_place: # NY
                        ba2ba_ny.append(pair_coref)
                    else: # YY
                        ba2ba_yy.append(pair_coref)
                else:
                    ba2no.append(pair_coref) # balance -> no balance
                    if not e_i_has_part and not e_i_has_place: # NN
                        ba2no_nn.append(pair_coref)
                    elif e_i_has_part and not e_i_has_place: # YN
                        ba2no_yn.append(pair_coref)
                    elif not e_i_has_part and e_i_has_place: # NY
                        ba2no_ny.append(pair_coref)
                    else: # YY
                        ba2no_yy.append(pair_coref)
            else: # no balance
                if e_i_new_part == e_j_new_part and e_i_new_place == e_j_new_place: # balance
                    no2ba.append(pair_coref) # no balance -> balance
                else:
                    no2no.append(pair_coref) # no balance -> no balance
        for pair_id, pair_results in pred_wrong_event_pairs.items():
            pair_coref = ['0', str(pair_results['coref'])]
            e_i_has_part, e_i_has_place = pair_results['e_i_has_part'], pair_results['e_i_has_place']
            e_j_has_part, e_j_has_place = pair_results['e_j_has_part'], pair_results['e_j_has_place']
            e_i_related_part, e_i_related_place = pair_results['e_i_related_part'], pair_results['e_i_related_place']
            e_j_related_part, e_j_related_place = pair_results['e_j_related_part'], pair_results['e_j_related_place']
            e_i_new_part, e_j_new_part = e_i_has_part or e_i_related_part, e_j_has_part or e_j_related_part
            e_i_new_place, e_j_new_place = e_i_has_place or e_i_related_place, e_j_has_place or e_j_related_place
            if e_i_has_part == e_j_has_part and e_i_has_place == e_j_has_place: # balance
                if e_i_new_part == e_j_new_part and e_i_new_place == e_j_new_place: # balance
                    ba2ba.append(pair_coref) # balance -> balance
                    if not e_i_has_part and not e_i_has_place: # NN
                        ba2ba_nn.append(pair_coref)
                    elif e_i_has_part and not e_i_has_place: # YN
                        ba2ba_yn.append(pair_coref)
                    elif not e_i_has_part and e_i_has_place: # NY
                        ba2ba_ny.append(pair_coref)
                    else: # YY
                        ba2ba_yy.append(pair_coref)
                else:
                    ba2no.append(pair_coref) # balance -> no balance
                    if not e_i_has_part and not e_i_has_place: # NN
                        ba2no_nn.append(pair_coref)
                    elif e_i_has_part and not e_i_has_place: # YN
                        ba2no_yn.append(pair_coref)
                    elif not e_i_has_part and e_i_has_place: # NY
                        ba2no_ny.append(pair_coref)
                    else: # YY
                        ba2no_yy.append(pair_coref)
            else: # no balance
                if e_i_new_part == e_j_new_part and e_i_new_place == e_j_new_place: # balance
                    no2ba.append(pair_coref) # no balance -> balance
                else:
                    no2no.append(pair_coref) # no balance -> no balance
    metrics = {}
    y_true, y_pred = [res[0] for res in ba2no], [res[1] for res in ba2no]
    metrics['balance -> no balance'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in ba2ba], [res[1] for res in ba2ba]
    metrics['balance -> balance'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in no2ba], [res[1] for res in no2ba]
    metrics['no balance -> balance'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in no2no], [res[1] for res in no2no]
    metrics['no balance -> no balance'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    ba2no_metrics = {}
    y_true, y_pred = [res[0] for res in ba2no_nn], [res[1] for res in ba2no_nn]
    ba2no_metrics['balance -> no balance [NN]'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in ba2no_yn], [res[1] for res in ba2no_yn]
    ba2no_metrics['balance -> no balance [YN]'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in ba2no_ny], [res[1] for res in ba2no_ny]
    ba2no_metrics['balance -> no balance [NY]'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    if len(ba2no_yy) > 0:
        y_true, y_pred = [res[0] for res in ba2no_yy], [res[1] for res in ba2no_yy]
        ba2no_metrics['balance -> no balance [YY]'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    ba2ba_metrics = {}
    y_true, y_pred = [res[0] for res in ba2ba_nn], [res[1] for res in ba2ba_nn]
    ba2ba_metrics['balance -> balance [NN]'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in ba2ba_yn], [res[1] for res in ba2ba_yn]
    ba2ba_metrics['balance -> balance [YN]'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in ba2ba_ny], [res[1] for res in ba2ba_ny]
    ba2ba_metrics['balance -> balance [NY]'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in ba2ba_yy], [res[1] for res in ba2ba_yy]
    ba2ba_metrics['balance -> balance [YY]'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    return {
        'metrics': metrics, 
        'ba2no_metrics': ba2no_metrics, 
        'ba2ba_metrics': ba2ba_metrics
    }

# gold_coref_file = '../../data/test.json'
# gold_simi_coref_file = '../../data/KnowledgeExtraction/simi_gold_test_related_info_0.75.json'
# pred_coref_file = '../clustering/event-event/epoch_5_dev_f1_72.9053_weights.bin_longformer_m_htao_hn_test_pred_corefs.json'
# pred_simi_coref_file = '../../data/KnowledgeExtraction/simi_epoch_3_test_related_info_0.75.json'
# metrics = different_info_balance_metrics(gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file)
# print(metrics['metrics'])
# print(metrics['ba2no_metrics'])
# print(metrics['ba2ba_metrics'])
