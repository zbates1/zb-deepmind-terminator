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

def perform_dimensionality_reduction(combined_embeddings):
    

    

# ============================================Main Function===============================================
def main(args):
    
    sorted_embed_files = extract_embedding_list(args.embed_file_path)
    # single_raw_embeddings = load_raw_embeddings(sorted_embed_files)
    
    combined_embeddings = concat_embeddings(sorted_embed_files, row_num=1000, col_num=-1)
    
    perform_dimensionality_reduction(combined_embeddings)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine embedding files")
    parser.add_argument("embed_file_path", type=str, help="The file path to the raw embeddings")
    args = parser.parse_args()
    main(args)