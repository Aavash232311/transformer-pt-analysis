''' Let's load the model with skeleton in same configuration '''
import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from script.shell import MinimalTransformer

''' For extreme reuseability make those hyperparams also paramaterised '''
def explained_variance(full_path):


    if not os.path.exists(full_path):
        sys.exit()    # re-run the train and see, this script is for analysis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 10

    ''' Important note we can't import the original transfoemr cause there, we have different hyperparamaster initlized. '''

    model = MinimalTransformer(vocab_size=vocab_size)
    model.load_state_dict(torch.load(full_path, map_location=device)) # loaded into transformer

    embedding_learned = model.token_embed # token embedding is vocab_size X d_model
    B, T = embedding_learned.weight.shape

    embedding_matrix = embedding_learned.weight.detach()

    X = embedding_matrix.cpu().numpy() 

    pca = PCA(n_components=8) 

    explained_variance_ratio = pca.explained_variance_ratio_

    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(8, 8))
    plt.plot(cumulative_variance_ratio, marker='o')

    for i, val in enumerate(cumulative_variance_ratio):
        plt.annotate(f'PC{i+1} {val:.3%} ', 
                    (i, val), 
                    textcoords="offset points", 
                    xytext=(0,8), 
                    ha='center')

    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio by Principal Components')

    plt.show()
