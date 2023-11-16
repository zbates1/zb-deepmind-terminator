# [1]: env LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/include/cudnn.h python3 [1]_auto_cmd_zbgpu.py --input_filenam term_train_sequences.txt

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3
import argparse
import sys
import gc
import haiku as hk
import jax
import jax.numpy as jnp
from . import get_pretrained_model
import pandas as pd
from jax import grad, jit, vmap

tf.config.experimental.set_visible_devices([], "GPU")
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".80"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"



cpus = jax.devices("cpu")
gpus = jax.devices("gpu")
print('JAX CPU and GPU: ', cpus, gpus)


#================ function definitions ======================================================
def process_sequences(parameters, forward_fn, token_ids):

    tokens = jnp.asarray(token_ids, dtype=jnp.int32)
    
    random_key = jax.random.PRNGKey(0)
    outs = forward_fn.apply(parameters, random_key, tokens)
    
    embeddings = outs["embeddings_20"][:, 1:, :]  # removing CLS token
    padding_mask = jnp.expand_dims(tokens[:, 1:] != tokenizer.pad_token_id, axis=-1)
    masked_embeddings = embeddings * padding_mask  # multiply by 0 pad tokens embeddings
    sequences_lengths = jnp.sum(padding_mask, axis=1)
    # mean_embeddings = jnp.sum(masked_embeddings, axis=1) / sequences_lengths
    
    return embeddings


# Process sequences and get mean embeddings
def batch_compiled_sequences(token_ids_array, batch_size=100):
    all_embeddings = []
    # all_mean_embeddings = []
    
    for i in range(0, len(token_ids_array), batch_size):
        print(i,  'Length Token IDs Array: ', len(token_ids_array))
        batch = token_ids_array[i : i + batch_size]
        # mean_embeddings, embeddings = compiled_process_sequences(batch)
        all_embeddings.append(embeddings)        
        print(i, len(all_embeddings))

        # all_mean_embeddings.append(mean_embeddings)
        
    return all_embeddings
#=================================================================


# Parse args
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process embeddings with dimensionality reduction methods")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of rows for reshaping embeddings")
    parser.add_argument("--starting_row", type=int, default=0, help="row to start")
    parser.add_argument("--input_filename", type=str, default = "./term_train_sequences.txt")
    parser.add_argument("--output_basename", type=str, default="embeddings_file.npy")
    args = parser.parse_args() 
    output_file_name = f'../data/ds_embeddings/{os.path.basename(args.input_filename).split(".")[0]}_start={args.starting_row}_batchsize={args.batch_size}_{args.output_basename}'
    # Make folder if it doesn't exist
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    print("output filename = ", output_file_name)

    print('Starting jax')


    # Instantiate the model
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name="2B5_multi_species",
        mixed_precision=False,
        embeddings_layers_to_save=(20,),
        max_positions=65, 
    )
    forward_fn = hk.transform(forward_fn)
    print('Passed forward_fn')


    client = jax.lib.xla_bridge.get_backend()

    try:
        sequences_df = pd.read_csv(args.input_filename, nrows=args.batch_size, skiprows=args.starting_row)
    # Except with specific error (file not found):
    except pd.errors.ParserError as e:
        print(e)


    # Get first column of sequence_df
    sequences = sequences_df.iloc[:, 0].tolist()
    
    assert len(sequences) == args.batch_size, f"Expected {args.batch_size} sequences, but got {len(sequences)}"

    # JIT compile the processing function
    compiled_process_sequences = jax.jit(lambda x: process_sequences(parameters, forward_fn, x))

    # Get token IDs from sequences using the tokenizer
    token_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
    print('Token IDs Created')
    # Convert token IDs to a JAX array
    token_ids_array = jnp.array(token_ids, dtype=jnp.int32)
        
    # mean_embeddings, embeddings = compiled_process_sequences(token_ids_array)
    embeddings, mean_embeddings = batch_compiled_sequences(token_ids_array, batch_size=100)

    jax.numpy.save(output_file_name, embeddings, allow_pickle=True, fix_imports=True)
    print(f'Embedding file index {args.starting_row} saved!')
    sys.exit(0)
