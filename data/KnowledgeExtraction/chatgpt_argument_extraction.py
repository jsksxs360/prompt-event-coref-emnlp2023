import openai

openai.api_key = "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"

def get_argument_info(pretty_event_mention, trigger, start_label, end_label):
    event = f'{start_label} {trigger} {end_label}'
    assert event in pretty_event_mention
    prefix = f'In the following document, please focus on the event {event}: '
    instruction = (
        f'Please extract the participants and locations of the event {event} '
        'in the above document. Respond in the form of a list:'
    )
    prompt = prefix + pretty_event_mention + '\n' + instruction
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=0
    )
    res = response['choices'][0]['message']['content'].strip()
    res_items = res.split('\n')
    flag, participants, locations, unknow = '', [], [], []
    for line in res_items:
        line = line[:line.find('(')].strip() if line.find('(') != -1 else line.strip()
        if not line:
            continue
        if not line.startswith('-'):
            if 'participant' in line.lower():
                flag = 'part'
            elif 'location' in line.lower():
                flag = 'loc'
            else:
                flag = 'unk'
        else:
            (participants if flag == 'part' else locations if flag == 'loc' else unknow).append(
                line[1:].strip()
            )
    return {
        'text': res, 
        'participants': participants, 
        'locations': locations, 
        'unknow': unknow
    }

em = '''
Pope Benedict XVI to resign due to "deteriorated strength" Pope Benedict XVI to resign due to "deteriorated strength" ROME, Feb. 11 (Xinhua) -- Pope Benedict XVI said during a meeting of Vatican cardinals on Monday that he will resign on Feb. 28. Speaking in Latin language, the 85-year-old Pope said that after deep reflection he has come to the certainty that his strengths, "due to an advanced age, are no longer suited to an adequate exercise" of his ministry in today's world, subject to rapid changes and questions of strong relevance. "Well aware of the seriousness of this act, with full freedom I [EVENT] declare [/EVENT] that I renounce the ministry of Bishop of Rome," he said adding that was quitting "for the good of the Church." He also announced that he would step down from the helm of the Catholic Church at 8:00 p.m. local time (1900 GMT) on Feb.28, leaving the papacy vacant so that "a Conclave to elect the new Supreme Pontiff will have to be convoked." He also said that he wishes to continue to serve the church "through a life dedicated to prayer". Upon resigning, he will move to the papal summer residence near Rome, and then will transfer to a cloistered residence in the Vatican. The pope's spokesman Federico Lombardi told a press conference soon after the announcement that Benedict will not take part in the conclave to elect his successor. The new pope could be elected before the end of March among several contenders in the wings.
'''

print(get_argument_info(em, 'declare', '[EVENT]', '[/EVENT]'))