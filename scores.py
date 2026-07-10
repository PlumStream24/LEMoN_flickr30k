import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, default=2, help='Index of the caption to retrieve scores for')
args = parser.parse_args()

def get_caption_scores(df, idx):
    print(f"Actual Caption : {df.loc[idx]['actual_label_text']}")
    print(f"Noisy Caption : {df.loc[idx]['noisy_label_text']}")
    print(f"Score: {df.loc[idx]['pred_score']}")
    print(f"Mislabelled: {df.loc[idx]['pred_mislabel']}")
    return

papers_best_hparams = {
    'tau_1_n': 0.274, 
    'tau_2_n': 0.074, 
    'tau_1_m': 0.072, 
    'tau_2_m': 0.0, 
    'beta': 0.092, 
    'gamma': 0.177
    }

df = pd.read_pickle('res.pkl')
get_caption_scores(df['df'], args.idx)