import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import spacy
import en_core_web_trf
from tqdm import tqdm

spacy.prefer_gpu()

# Paths to the Flickr30k karpathy split and image directory
meta_path = Path('dataset_flickr30k.json')
root_dir = Path('/flickr30k_images')

# Load the spaCy model
nlp = spacy.load("en_core_web_trf")
nlp = en_core_web_trf.load()

# Load the Flickr30k kartpathy split
meta = pd.read_json(meta_path)
meta = pd.DataFrame.from_records(meta['images'].values).set_index('imgid')
np.random.seed(42)

# Choose one caption per image at random from the available captions
meta['sentence'] = meta['sentences'].apply(lambda x: np.random.choice(x)['raw'])
# Parse the selected captions with spaCy
meta['spacy_doc'] = [d for d in tqdm(nlp.pipe(meta['sentence'], n_process = 1), total = len(meta))]


def extract_nouns(x):
    # Keep only noun tokens in lowercase
    return [i.text.lower().strip() for i in x if i.pos_ == 'NOUN']

def extract_tokens(x):
    # Keep all tokens as text
    return [i.text for i in x]

# Extract nouns and tokenize words from dataset captions
meta['nouns'] = meta['spacy_doc'].apply(extract_nouns)
meta['tokens'] = meta['spacy_doc'].apply(extract_tokens)

# Build a vocabulary of nouns and map each noun to a numeric ID
noun_vocab = tuple(set([j for i in meta['nouns'] for j in i]))
noun_vocab_mapping = {i: c for c, i in enumerate(noun_vocab)}

# Map nouns to integers so overlap-based computations are faster later
meta['nouns_int'] = meta['nouns'].apply(lambda x: [noun_vocab_mapping[i] for i in x])

# Save the processed split + spacy PoS tags
meta.drop(columns = ['spacy_doc']).to_pickle('multimodal_mislabel_split.pkl')