import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, help='Index of the caption to retrieve scores for')
parser.add_argument('--hparams', action='store_true', default=False, help='Print Hyperparameters')
args = parser.parse_args()

def get_caption_scores(df, idx):
    print(f"Actual Caption : {df.loc[idx]['actual_label_text']}")
    print(f"Noisy Caption : {df.loc[idx]['noisy_label_text']}")
    print(f"Score: {df.loc[idx]['pred_score']}")
    print(f"Mislabel pred: {df.loc[idx]['pred_mislabel']}")
    return

def get_params(hparams):
    print(f'beta: {hparams["beta"]}')
    print(f'gamma: {hparams["gamma"]}')
    print(f'tau_1_n: {hparams["tau_1_n"]}')
    print(f'tau_2_n: {hparams["tau_2_n"]}')
    print(f'tau_1_m: {hparams["tau_1_m"]}')
    print(f'tau_2_m: {hparams["tau_2_m"]}')
    print(f'threshold: {hparams["thres"]}')
    print(f'validation f1: {hparams["val_f1"]}')

papers_best_hparams = {
    'tau_1_n': 0.274, 
    'tau_2_n': 0.074, 
    'tau_1_m': 0.072, 
    'tau_2_m': 0.0, 
    'beta': 0.092, 
    'gamma': 0.177
    }

df = pd.read_pickle('./dump/res.pkl')
if args.idx is not None:
    get_caption_scores(df['df'], args.idx)
if args.hparams:
    get_params(df['hparams'])
