import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import faiss
import tqdm
import pickle
import pandas as pd

from transformers import AutoTokenizer, CLIPModel
from torch.utils.data import DataLoader

from own.utils_data import normalize_vectors
from own.utils_data import get_captioning_dataset
from own.utils_metric import calc_scores_given_hparams_vectorized
from own.utils_metric import optimize_f1_efficient
from own.utils_metric import maximize_metric

parser = argparse.ArgumentParser(description="LEMoN")
parser.add_argument('--knn_k', default = 5, type = int)
parser.add_argument('--batch_size', default = 128, type = int)
parser.add_argument("--dist_type", type=str, default="cosine", choices=["cosine", "euclidean"])
parser.add_argument('--compr_dataset_size_limit', default = 50000, type = int)
#parser.add_argument('--skip_train', action = 'store_true')
#parser.add_argument('--skip_hparam_optim', action = 'store_true')

args = parser.parse_args()
bs = args.batch_size
k = args.knn_k
dist_type = args.dist_type
compr_dataset_size_limit = args.compr_dataset_size_limit
skip_train = True
#skip_train = args.skip_train
#skip_hparam_optim = args.skip_hparam_optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set, val_set, test_set = get_captioning_dataset(name = 'flickr30k', data_seed = 42, percent_flips = 0.3, flip_type = 'random', data_transform = None, cluster = False)

tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32')
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
model = model.to(device)
model.eval()

if len(train_set) > compr_dataset_size_limit:
    train_indices_in_compr = np.random.choice(np.arange(len(train_set)), compr_dataset_size_limit, replace = False)
    compr_set = torch.utils.data.Subset(train_set, train_indices_in_compr)
else:
    train_indices_in_compr = np.arange(len(train_set))
    compr_set = train_set

dataloader = DataLoader(dataset=compr_set, batch_size=bs, num_workers=4)

start_t = datetime.now()
emb_img, emb_txt, tr_text_labels = [], [], []
for idx, batch in enumerate(dataloader):
    pixel_values = batch[0].to(device)

    text_labels = batch[2]    
    tr_text_labels += text_labels
    
    encodings = tokenizer(text_labels, padding="max_length", truncation=True)
    input_ids = torch.tensor(encodings["input_ids"]).to(device)
    attention_mask = torch.tensor(encodings["attention_mask"]).to(device)

    with torch.no_grad():
        emb_txt.append(model.get_text_features(input_ids, attention_mask))
        emb_img.append(model.get_image_features(pixel_values))

emb_txt_tr = normalize_vectors(torch.concat(emb_txt))
emb_img_tr = normalize_vectors(torch.concat(emb_img))

if dist_type == 'cosine':
    index_txt = faiss.IndexFlatIP(emb_txt_tr.shape[1])
    index_img = faiss.IndexFlatIP(emb_img_tr.shape[1])
    dists_tr = 1 - (emb_txt_tr * emb_img_tr).sum(dim = 1)
elif dist_type == 'euclidean':
    index_txt = faiss.IndexFlatL2(emb_txt_tr.shape[1])
    index_img = faiss.IndexFlatL2(emb_img_tr.shape[1])
    dists_tr = ((emb_txt_tr - emb_img_tr)**2).sum(dim = 1)

index_txt.add(emb_txt_tr.numpy())
index_img.add(emb_img_tr.numpy())
tr_text_labels = np.array(tr_text_labels)

logs = []
if skip_train:
    sets_iter = zip(['val', 'test'],  [val_set, test_set])
else:
    sets_iter = zip(['train', 'val', 'test'], [train_set, val_set, test_set])

for sname, dset in sets_iter:
    dataloader = DataLoader(
        dataset=dset, batch_size=bs, num_workers=4
    )
    for idx, batch in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):
        noisy_labels = batch[2]
        real_labels = batch[1]
        pixel_values = batch[0].to(device)

        noisy_text_labels = noisy_labels
        noisy_text_labels_prompts = noisy_labels
        clean_text_labels = real_labels
                
        label_flips = np.array(noisy_text_labels)==np.array(clean_text_labels)
        label_flips = 1-label_flips
        
        encodings = tokenizer(
                noisy_text_labels_prompts, padding="max_length", truncation=True)
        
        input_ids = torch.tensor(encodings["input_ids"]).to(device)
        attention_mask = torch.tensor(encodings["attention_mask"]).to(device)

        with torch.no_grad():
            text_embeds = normalize_vectors(model.get_text_features(input_ids, attention_mask))
            img_embeds = normalize_vectors(model.get_image_features(pixel_values))

        D_ns, I_ns = index_img.search(img_embeds.numpy(), k + (sname == 'train'))
        D_ms, I_ms = index_txt.search(text_embeds.numpy(), k+ (sname == 'train'))

        for i in range(len(img_embeds)):
            sample_idx = idx * bs + i
            img_embed = img_embeds[i, None]
            text_embed = text_embeds[i, None]

            if dist_type == 'cosine':
                d1 = 1 - torch.dot(img_embed.flatten(), text_embed.flatten())
            elif dist_type == 'euclidean':
                d1 = ((img_embed.flatten() - text_embed.flatten())**2).sum()

            # d_n
            D_n, I_n = D_ns[i], I_ns[i]
            if sname == 'train': # skip over same sample
                if sample_idx in train_indices_in_compr:
                    I_n = I_n[1:]
                    D_n = D_n[1:]
                else:
                    I_n = I_n[:-1]
                    D_n = D_n[:-1]
            y_n = emb_txt_tr[I_n]

            if dist_type == 'cosine':
                D_n = -D_n
                dists_n = 1 - (text_embed * y_n).sum(dim = 1)
            elif dist_type == 'euclidean':
                dists_n = ((text_embed - y_n)**2).sum(dim = 1)

            # d_m
            D_m, I_m = D_ms[i], I_ms[i]
            if sname == 'train': # skip over same sample
                if sample_idx in train_indices_in_compr:
                    I_m = I_m[1:]
                    D_m = D_m[1:]
                else:
                    I_m = I_m[:-1]
                    D_m = D_m[:-1]
            x_m = emb_img_tr[I_m]
            if dist_type == 'cosine':
                D_m = -D_m
                dists_m = 1 - (img_embed * x_m).sum(dim = 1)
            elif dist_type == 'euclidean':
                dists_m = ((img_embed - x_m)**2).sum(dim = 1)

            logs.append({
                'sset': sname,
                'idx': sample_idx,
                'actual_label': real_labels[i].item() if torch.is_tensor(real_labels[i]) else real_labels[i],
                'actual_label_text': clean_text_labels[i],
                'noisy_label': noisy_labels[i],
                'noisy_label_text': noisy_text_labels[i],
                'is_mislabel': label_flips[i],
                'is_correct_label': 1 - label_flips[i],
                'd_1': d1.item(),
                'dists_n': dists_n.numpy(),
                'D_n': D_n.flatten(),
                'dists_tr_n': dists_tr[I_n].numpy(),
                'dists_m': dists_m.numpy(),
                'D_m': D_m.flatten(),
                'dists_tr_m': dists_tr[I_m].numpy()
            })

end_t = datetime.now()
timedelta = (end_t - start_t).total_seconds() 
n_samples = len(logs)
print(f"Finished {n_samples} samples in {timedelta} seconds; avg of {timedelta/n_samples}s per sample")

df = pd.DataFrame(logs)

df_val = df.query('sset == "val"')

grid = {
    'beta': np.arange(0, 100.01, 5),
    'gamma': np.arange(0, 100.01, 5),
    'tau_1': [0, 1, 5, 10],
    'tau_2': [0, 1, 5, 10],
}

# --- Fit hyperparameters on validation set ---
(best_beta, best_gamma, best_tau_1_n, best_tau_2_n, best_tau_1_m, best_tau_2_m), best_f1, best_thres = maximize_metric(
    df_val,
    grid,
    [[0] * 6, [0.5] * 6, [1] * 6, [10] * 6],
    optimize_f1_efficient,
    {},                 # side_info / obj_func_args — empty for optimize_f1_efficient
    force_zero=[],
    force_one=[]
)

hparams = {
    'beta': best_beta,
    'gamma': best_gamma,
    'tau_1_n': best_tau_1_n,
    'tau_2_n': best_tau_2_n,
    'tau_1_m': best_tau_1_m,
    'tau_2_m': best_tau_2_m,
}
# -----------------------------------------------

# --- Apply fitted hyperparameters to classify every sample ---
df['pred_score'], df['d_n'], df['d_m'] = calc_scores_given_hparams_vectorized(
    df, hparams, True
)
df['pred_mislabel'] = df['pred_score'] >= best_thres
# ---------------------------------------------------------------

res = {
    'df': df,
    'hparams': hparams,
    'thres': best_thres,
    'val_f1': best_f1,
}

pickle.dump(res, Path('res.pkl').open('wb'))