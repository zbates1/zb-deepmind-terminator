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

from .tools.embeddings_generator import 

def run_embeddings_generator(args, num_batches):
    for i in range(0,num_batches):
        args.starting_row = i*args.batch_size
        print('Starting Row: ', args.starting_row)
        # Just use the cudnn library path in this function: 
        # env LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/include/cudnn.h python3 [1]_auto_cmd_zbgpu.py
        cmd = f'python3 ./tools/embeddings_generator.py --batch_size={args.batch_size} --starting_row={args.starting_row} --input_filename={args.input_filename}'
        print(f"running cmd: {cmd}")
        os.system(cmd)
        
def main():
    args = parser.parse_args()
    embeddings_generator(args, num_batches)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Automatically run embeddings_generator.py for whole dataset")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of rows for reshaping embeddings")
    parser.add_argument("--starting_row", type=int, default=0, help="Row to begin")
    parser.add_argument("--input_filename", type=str, default = "./data/shalem.txt", help="input filename")
    parser.add_argument("--output_basename", type=str, default="embeddings_file.npy", help="output basename, default is embeddings_file.npy")
    # "env LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/include/cudnn.h "
    # This may not be needed for every user, however it is for my instance. I need to pass the cudnn library path
    parser.add_argument("--add_cudnn", type=str, default="", help="add cudnn library path, like this: env LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/include/cudnn.h")
    
    args = parser.parse_args()

    sequence_file = pd.read_csv(f'{args.input_filename}', sep="\t", header=None)
    print('Length of Sequence File: ', len(sequence_file))
    
    remaining_after_batch = len(sequence_file)%args.batch_size
    print('Number of Batches: ', remaining_after_batch)
    
    num_batches = len(sequence_file)//args.batch_size
    print('Number of Batches: ', num_batches)

    main()
