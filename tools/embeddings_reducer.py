import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import glob
import sys
import re
import argparse
import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_embedding_list(embed_file_path):
    embed_files = glob.glob(f"{embed_file_path}/start_*.npy") 
    print(f"Number of npy files in {embed_file_path}: {len(embed_files)}")

    sorted_files = sorted(embed_files, key=extract_number)
    print('\nSorted Files:', sorted_files)
    return sorted_files

def extract_number(path):
    match = re.search(r'(\d+)', path)
    return int(match.group(1)) if match else -1

def load_raw_embeddings(filepath):
    print('\nLoading Embeddings from:', filepath)
    raw_embeddings = np.load(filepath)
    return raw_embeddings

def embeddings_reformat(raw_embeddings, reshape_rows, reshape_cols):
    list_embeddings = raw_embeddings.tolist()
    embeddings_array = np.array(list_embeddings)
    print('\nEmbeddings Array Shape:', embeddings_array.shape)
    reshaped_embeddings = embeddings_array.reshape(args.reshape_rows, args.reshape_cols)
    print('\nReshaped Embeddings Shape:', reshaped_embeddings.shape)
    return reshaped_embeddings

def concat_embeddings(sorted_embed_files, row_num, col_num):

    all_embeddings = []

    for filepath in tqdm.tqdm(sorted_embed_files, desc="Processing files"):
        print('\nLoading Embeddings from:', filepath)
        raw_embeddings = load_raw_embeddings(filepath)  # Assuming this returns a NumPy array
        reshaped_embeddings = embeddings_reformat(raw_embeddings, row_num, col_num)
        print('\nReshaped Embeddings Shape:', reshaped_embeddings.shape)
        all_embeddings.append(reshaped_embeddings)

    combined_embeddings = np.vstack(all_embeddings)
    print('\nCombined Embeddings Shape:', combined_embeddings.shape)
    assert not np.isnan(combined_embeddings).any()
    return combined_embeddings

def perform_dimensionality_reduction(combined_embeddings, methods, components, base_dir, overwrite):

    for method in methods:
        for num_components in components:
            print('Method and number of components:', method, num_components)

            if method == 'pca':
                file_name = os.path.join(base_dir, f'pca_components_{num_components}.npy')

            else:
                print(f"Unsupported method: {method}")
                continue

            if os.path.exists(file_name) and not overwrite:
                print(f"File {file_name} already exists. Skipping.")
                continue

            X_reshaped_scaled = StandardScaler().fit_transform(combined_embeddings)

            if method == 'pca':
                dim_reduct = PCA(n_components=num_components)
            else:
                print(f"Unsupported method: {method}! If you would like to add this capability, check the perform_dimensionality_reduction function, and pass the string into the methods argument.")
                continue
            
            X_reduced = dim_reduct.fit_transform(X_reshaped_scaled)
            assert X_reduced.shape[1] == num_components, f'Expected {num_components} components, but got {X_reduced.shape[1]}'

            print('Shape of X array after reduction:', X_reduced.shape)
            assert not np.isnan(X_reduced).any(), f'The numpy array contains NaNs'
            
            np.save(file_name, X_reduced)
            print(f"Reduced data saved to {file_name}")


    

# ============================================Main Function===============================================
def process_embeddings(embed_folder_path):
    
    sorted_embed_files = extract_embedding_list(embed_folder_path)
    # single_raw_embeddings = load_raw_embeddings(sorted_embed_files)
    
    combined_embeddings = concat_embeddings(sorted_embed_files, row_num=1000, col_num=-1)
    
    
    level_one_path_abstraction = os.path.dirname(embed_folder_path)
    file_path = os.path.join(os.path.dirname(level_one_path_abstraction), 'reduced_embeddings')
    os.makedirs(file_path, exist_ok=True)
    print('\nFile Path to save reduced embeddings:', file_path)
    
    perform_dimensionality_reduction(combined_embeddings, methods=['pca'], components=[2, 5, 10, 150], base_dir=file_path, overwrite=False)
    
    

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Combine embedding files")
    # parser.add_argument("embed_file_path", type=str, help="The file path to the raw embeddings")
    # args = parser.parse_args()
    # main(args)
    
    process_embeddings('./data/ds_embeddings')