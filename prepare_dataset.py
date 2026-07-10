import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import spacy
from tqdm import tqdm

spacy.prefer_gpu()

meta_path = Path('/kaggle/input/datasets/shtvkumar/karpathy-splits/dataset_flickr30k.json')
root_dir = Path('/kaggle/input/datasets/hsankesara/flickr-image-dataset/flickr30k_images')

nlp = spacy.load("en_core_web_trf")
import en_core_web_trf
nlp = en_core_web_trf.load()
meta = pd.read_json(meta_path)
meta = pd.DataFrame.from_records(meta['images'].values).set_index('imgid')
np.random.seed(42)
meta['sentence'] = meta['sentences'].apply(lambda x: np.random.choice(x)['raw'])
meta['spacy_doc'] = [d for d in tqdm(nlp.pipe(meta['sentence'], n_process = 1), total = len(meta))]
def extract_nouns(x):
    return [i.text.lower().strip() for i in x if i.pos_ == 'NOUN']

def extract_tokens(x):
    return [i.text for i in x]

meta['nouns'] = meta['spacy_doc'].apply(extract_nouns)
meta['tokens'] = meta['spacy_doc'].apply(extract_tokens)
noun_vocab = tuple(set([j for i in meta['nouns'] for j in i]))
noun_vocab_mapping = {i: c for c, i in enumerate(noun_vocab)}
# map nouns to integers; will allow us to compute overlap faster later
meta['nouns_int'] = meta['nouns'].apply(lambda x: [noun_vocab_mapping[i] for i in x])
meta.drop(columns = ['spacy_doc']).to_pickle('multimodal_mislabel_split.pkl')