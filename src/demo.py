from corefprompt import CorefPrompt

model_checkpoint='../PT_MODELS/roberta-large/'
best_weights='./epoch_2_dev_f1_71.5876_weights.bin'
coref_model = CorefPrompt(model_checkpoint, best_weights)

document = 'Former Pakistani dancing girl commits suicide 12 years after horrific acid attack which left her looking "not human". She had undergone 39 separate surgeries to repair damage. Leapt to her death from sixth floor Rome building earlier this month. Her ex-husband was charged with attempted murder in 2002 but has since been acquitted.'

ev1 = {
    'offset': 38, 
    'trigger': 'suicide', 
    'args': [
        {'mention': 'Former Pakistani dancing girl', 'role': 'participant'}
    ]
}
ev3 = {
    'offset': 88, 
    'trigger': 'left', 
    'args': [
        {'mention': 'acid', 'role': 'participant'}, 
        {'mention': 'her', 'role': 'participant'}
    ]
}
ev4 = {
    'offset': 168, 
    'trigger': 'damage', 
    'args': [
        {'mention': 'She', 'role': 'participant'}
    ]
}
ev5 = {
    'offset': 189, 
    'trigger': 'death', 
    'args': [
        {'mention': 'her', 'role': 'participant'}, 
        {'mention': 'sixth floor Rome building', 'role': 'place'}
    ]
}

# direct predict event pairs
res = coref_model.predict_coref_in_doc(document, ev1, ev5)
print('[Prompt]:', res['prompt'])
print(f"ev1[{ev1['trigger']}] - ev5[{ev5['trigger']}]: {res['label']} ({res['probability']})")

# # predict event pairs in the same document
# coref_model.init_document(document)
# res = coref_model.predict_coref(ev1, ev5)
# print(f"ev1[{ev1['trigger']}] - ev5[{ev5['trigger']}]: {res['label']} ({res['probability']})")
# res = coref_model.predict_coref(ev1, ev3)
# print(f"ev1[{ev1['trigger']}] - ev3[{ev3['trigger']}]: {res['label']} ({res['probability']})")
# res = coref_model.predict_coref(ev1, ev4)
# print(f"ev1[{ev1['trigger']}] - ev4[{ev4['trigger']}]: {res['label']} ({res['probability']})")
# res = coref_model.predict_coref(ev3, ev4)
# print(f"ev3[{ev3['trigger']}] - ev4[{ev4['trigger']}]: {res['label']} ({res['probability']})")
