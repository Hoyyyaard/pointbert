import os
import json
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--f', type=str)
args = argparser.parse_args()

pred = json.load(open(args.f, 'r'))
corpus = json.load(open('data/NuscenesQA/questions/NuScenes_val_questions.json', 'r'))['questions']

val_corpus = {
    'exist_0': [],
    'exist_1': [],
    'exist_All': [],
    'object_0': [],
    'object_1': [],
    'object_All': [],
    'status_0': [],
    'status_1': [],
    'status_All': [],
    'count_0': [],
    'count_1': [],
    'count_All': [],
    'comparison_0': [],
    'comparison_1': [],
    'comparison_All': [],
    'All': []

}

for cor in corpus:
    id = '{}#{}#{}'.format(cor["sample_token"], cor['question'], cor['answer'])
    if id in pred:
        acc = int(pred[id]['pred'][0] == cor['answer'])
        val_corpus['All'].append(acc)
        val_corpus['{}_{}'.format(cor['template_type'], cor['num_hop'])].append(acc)
        val_corpus['{}_All'.format(cor['template_type'])].append(acc)

# Print to scene
for key, v in val_corpus.items():
    print('{}: {}'.format(key, sum(v)/len(v)))