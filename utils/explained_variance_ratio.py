''' Let's load the model with skeleton in same configuration '''
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

''' For extreme reuseability make those hyperparams also paramaterised '''

def explained_variance(full_path, model, pc):


    if not os.path.exists(full_path):
        sys.exit()    # re-run the train and see, this script is for analysis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ''' Important note we can't import the original transfoemr cause there, we have different hyperparamaster initlized. '''
    model.load_state_dict(torch.load(full_path, map_location=device)) # loaded into transformer
    embedding_learned = model.token_embed # token embedding is vocab_size X d_model
    B, T = embedding_learned.weight.shape

    embedding_matrix = embedding_learned.weight.detach()

    X = embedding_matrix.cpu().numpy() 

    pca = PCA(n_components=pc) 

    pca.fit(X)


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


def overfitting_plot(train_history, eval_losses):
    epochs = [x['epoch'] for x in train_history]
    train_losses = [x['loss'] for x in train_history]
    
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_losses, label='Training Loss', color='red', linewidth=2)
    plt.plot(epochs, eval_losses, label='Evaluation Loss', color='orange', linewidth=2)

    plt.title('Overfitting Check: Training vs Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
