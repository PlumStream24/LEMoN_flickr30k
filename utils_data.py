from pathlib import Path

import torch
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import pandas as pd
import numpy as np

### --- Static variables ---
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
flickr30k_path = './flickr30k_images'
generic_transform = transform = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ]
)

### Dataset class for captioning dataset
class CaptioningDataset(Dataset):
    def __init__(self, df, transform, dataset_name, use_cluster):
        self.df = df
        self.transform = transform
        self.dataset_name =  dataset_name
        self.use_cluster = use_cluster
    
    def __len__(self):
        return len(self.df)

    def get_image(self, x):
        if self.dataset_name == 'mimiccxr_caption':
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-5] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
            if reduced_img_path.is_file():
                return Image.open(reduced_img_path).convert("RGB")       
        return Image.open(x).convert("RGB")
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']

        if self.use_cluster:
            noisy_label = row['sent_cluster']
            real_label = -1 if row['is_mislabel'] else row['sent_cluster']
        else:
            real_label = row['gold_sentence']
            noisy_label = row['sentence']

        img = self.get_image(path)
        return self.transform(img), real_label, noisy_label
    


def build_index(arr):
    max_val = max([max(sublist) for sublist in arr if len(sublist) > 0]) + 1
    d = len(arr)
    arr_set = [set(arr[i]) for i in range(len(arr))]
    idx = {}
    
    for i in range(max_val):
        idx[i] = [c for c, sublist in enumerate(arr_set) if i in sublist]
    return idx   

# Create dictionary mapping {image index: index of new label}
# only for indices where the label is to be changed
def calc_noise_by_integer_matching(cat_labels, frac_noise = 0.3, seed = 42):
    index = build_index(cat_labels)
    rng = np.random.default_rng(seed)
    cand_idxs = np.arange(len(cat_labels)) 
    cand_idxs = [i for i in cand_idxs if len(cat_labels[i]) > 0] # can't match items with no categories
    
    to_change_idxs = rng.choice(cand_idxs, int(frac_noise * len(cat_labels)),
                                replace = False)
    change_dict = {}
    for i in to_change_idxs:
        choose_obj = rng.choice(cat_labels[i])
        subset = index[choose_obj]
        subset = np.setdiff1d(subset, [i])
        if len(subset) > 0: 
            change_dict[i] = rng.choice(subset, 1)[0]
    
    return change_dict

# Make random noise dictionary mapping {index of sample: index of new label}
# for a given fraction of the dataset.
def random_noise_dict(num_items, frac_noise = 0.4, seed = 42):
    rng = np.random.default_rng(seed)
    to_change_idxs = rng.choice( np.arange(num_items), int(frac_noise * num_items),
                                replace = False)
    change_dict = {}
    for i in to_change_idxs:
        change_dict[i] = rng.choice(np.delete(np.arange(num_items), i) # avoid matching to self
                                    , 1)[0]
    return change_dict

# Make noise dataset by swapping labels according to a dictionary mapping {index of sample: index of new label}
def noise_given_dict(meta, d):
    meta_c = meta.copy()
    meta_c['gold_sentence'] = meta_c['sentence']
    #get indeces for noise
    source_idx = meta.index[list(d.keys())]
    target_idx = meta.index[list(d.values())]

    meta_c.loc[source_idx, 'sentence'] = meta.loc[target_idx, 'sentence'].values    #swap the sentences for the mislabelled samples
    meta_c['is_mislabel'] = (meta_c['sentence'] != meta_c['gold_sentence'])         #set mislabel flag

    return meta_c



def get_captioning_dataset(name, data_seed, percent_flips=0.4, flip_type='random', data_transform=None, cluster = False):
    assert 0 <= percent_flips <= 1
    df = pd.read_pickle('multimodal_mislabel_split.pkl')
    if name == 'flickr30k':
        df['path'] = df.apply(lambda x: Path(flickr30k_path)/'flickr30k_images'/x['filename'], axis = 1)

    dfs = {}
    for split in ['train', 'val', 'test']:
        df_split = df.query(f'split == "{split}"')    
        if flip_type == 'random':
            noise_dict = random_noise_dict(len(df_split), percent_flips, data_seed)
        elif flip_type == 'noun':
            noise_dict = calc_noise_by_integer_matching(df_split['nouns_int'].values, percent_flips, data_seed)
        else:
            raise ValueError(f'Unsupported flip_type: {flip_type}')

        dfs[split] = noise_given_dict(df_split, noise_dict)
    
    if data_transform is None:
        data_transform = generic_transform
    return (CaptioningDataset(dfs['train'], data_transform, name, cluster), 
            CaptioningDataset(dfs['val'], data_transform, name, cluster), 
            CaptioningDataset(dfs['test'], data_transform, name, cluster)
    )

def normalize_vectors(vectors):
    return torch.nn.functional.normalize(vectors, p=2, dim=1)
