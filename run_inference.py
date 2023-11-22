import os
import argparse
import glob
import sys
import re
import pandas as pd

import nucleotide_transformer
from nucleotide_transformer.pretrained import get_pretrained_model
from tools.embeddings_reducer import process_embeddings
from tools.create_embeddings_dirname import create_ds_embeds_paths
from tools.load_model import load_nt_model
from tools.embeddings_generator import main as embeddings_generation

def run_embeddings_generator(args, num_batches, parameters, forward_fn, tokenizer, config, num_existing_files):
    for i in range(0,num_batches):
        if i < num_existing_files:
            print(f'File {i} of {num_batches} already exists, skipping...')
            continue
        else:
            print(f'Processing File {i} of {num_batches}...')
            args.starting_row = i*args.batch_size
            print('Starting Row: ', args.starting_row)

        embeddings_generation(args, num_batches, parameters, forward_fn, tokenizer, config)
        
def process_dataset(file_path, batch_size):
    seq_ds = pd.read_csv(file_path, header=None)
    
    num_batches = len(seq_ds)//batch_size
    print('Number of Batches: ', num_batches)
    
    return seq_ds, num_batches
        
def main_wrapper(args, num_batches, num_existing_files):
    print('Loading Model...')
    parameters, forward_fn, tokenizer, config = load_nt_model(model_name="2B5_multi_species", embeddings_layers_to_save=(20,), max_positions=65)
    
    print('Running Embeddings Generator...')
    run_embeddings_generator(args, num_batches, parameters, forward_fn, tokenizer, config, num_existing_files)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Automatically run embeddings_generator.py for whole dataset")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of rows for reshaping embeddings")
    parser.add_argument("--starting_row", type=int, default=0, help="Row to begin")
    parser.add_argument("--input_filename", type=str, default = "./data/shalem_15.txt", help="input filename")
    parser.add_argument("--output_basename", type=str, default="embeddings_file.npy", help="output basename, default is embeddings_file.npy")
    parser.add_argument("--add_cudnn", type=str, default="", help="add cudnn library path, like this: env LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/include/cudnn.h")
    
    args = parser.parse_args()
    
    seq_ds, num_batches = process_dataset(args.input_filename, args.batch_size)
    print(f'Head of seq_ds: {seq_ds.head()}')
    print(f'Number of batches: {num_batches}')


    # Gather the path names to the raw embeddings and processed embeddings to process into inputs for following functions
    single_ds_embed_path , embed_dirname = create_ds_embeds_paths(input_filename=args.input_filename, args=args)
    raw_embed_dirname = os.path.dirname(single_ds_embed_path)
    num_existing_raw_embed_files = len(glob.glob(raw_embed_dirname + '/*.npy'))

    print(f'Number of existing files in {embed_dirname}: {num_existing_raw_embed_files}')
    processed_embeds_dirname = os.path.join(embed_dirname, 'processed')
    print('Embedding Dirname: ', embed_dirname)
    
    print('Running main wrapper function...')
    main_wrapper(args, num_batches=num_batches, num_existing_files=num_existing_raw_embed_files)
    
    # Lets get the directory name
    print('Finished running embeddings generator, now reducing embeddings...')

    process_embeddings(raw_embed_dirname, processed_embeds_dirname)