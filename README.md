# LEMoN_flickr30k

### Step 0: Environment and Prerequisites

Run `conda env create -f environment.yml \ conda activate lemon`.

Alternatively if you run into dependency issues, you can just run the [kaggle notebook](https://www.kaggle.com/code/iqbalsigid/lemon-flickr30k).

### Step 1: Preprocessing Data

1. Download the Flickr30k images from [here](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset). Put `flickr30k_images` in the same directory as `preprocess_dataset.py`.

2. Karpathy split for Flickr30k is already available in the repository.

3. Run `preprocess_dataset.py`.

4. This will output `multimodal_mislabel_split` in the same directory.

### Step 2: Run Flickr30k Experiment

Use the below commands to run the experiment.
```
python lemon.py \
--knn_k 5                  \ number of k for nearest neighbor
--batch_size 128           \ batch size
--dist_type 'cosine'       \ 'cosine' | 'euclidean'
--noise_type 'noun'        \ 'random' | 'noun'
--noise_level 0.4          \ 0.0-1.0 range
--skip_train                 type in to skip train split in the calculation
```
If you want to run the default config just run `python lemon.py`.
This will output `res.pkl` which has the hyperparameter and scores of each samples, and scores.csv which is the scores in csv format.

### Additional Step
If you want to see score for each sample and its prediction run the following.
```
python scores.py \
--idx 0         \ samples index to see caption and prediction result
--hparam        \ to show hyperparameters
```