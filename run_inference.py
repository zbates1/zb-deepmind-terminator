import os
import argparse
import glob
import sys
import re
import argparse
import pandas as pd

import nucleotide_transformer
from nucleotide_transformer.pretrained import get_pretrained_model
from tools.embeddings_reducer import process_embeddings
from tools.create_embeddings_dirname import create_ds_embeds_paths

def run_embeddings_generator(args, num_batches):
    for i in range(0,num_batches):
        args.starting_row = i*args.batch_size
        print('Starting Row: ', args.starting_row)
        # Just use the cudnn library path in this function: 
        # env LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/include/cudnn.h python3 [1]_auto_cmd_zbgpu.py
        cmd = f'python3 ./tools/embeddings_generator.py --batch_size={args.batch_size} --starting_row={args.starting_row} --input_filename={args.input_filename}'
        print(f"running cmd: {cmd}")
        os.system(cmd)
        
def process_dataset(file_path, batch_size):
    seq_ds = pd.read_csv(file_path, header=None)
    
    num_batches = len(seq_ds)//batch_size
    print('Number of Batches: ', num_batches)
    
    return seq_ds, num_batches
        
def main_wrapper(args, num_batches):
    run_embeddings_generator(args, num_batches)
        
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

    main_wrapper(args, num_batches=num_batches)
    
    # Lets get the directory name
    _ , embed_dirname = create_ds_embeds_paths(input_filename=args.input_filename, args=args)
    processed_embeds_dirname = os.path.join(embed_dirname, 'processed')
    
    process_embeddings(processed_embeds_dirname)
    
    
