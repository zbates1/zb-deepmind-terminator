# embeddings_generator.py

import os
import argparse
import sys
import gc
import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
from jax import grad, jit, vmap
import jax.numpy as jnp
from jax.interpreters import xla
import tensorflow as tf

from tools.create_embeddings_dirname import create_ds_embeds_paths

import nucleotide_transformer
from nucleotide_transformer.pretrained import get_pretrained_model # This doesn't seem to work 11/16/23

tf.config.experimental.set_visible_devices([], "GPU")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".80"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

cpus = jax.devices("cpu")
gpus = jax.devices("gpu")
print('JAX CPU and GPU: ', cpus, gpus)


#================ function definitions ======================================================
def process_sequences(parameters, forward_fn, token_ids, tokenizer, embeddings_layer_to_save):

    tokens = jnp.asarray(token_ids, dtype=jnp.int32)
    
    random_key = jax.random.PRNGKey(0)
    outs = forward_fn.apply(parameters, random_key, tokens)
    
    embeddings = outs[f"embeddings_{embeddings_layer_to_save}"][:, 1:, :]  # removing CLS token
    padding_mask = jnp.expand_dims(tokens[:, 1:] != tokenizer.pad_token_id, axis=-1)
    masked_embeddings = embeddings * padding_mask  # multiply by 0 pad tokens embeddings
    sequences_lengths = jnp.sum(padding_mask, axis=1)
    # mean_embeddings = jnp.sum(masked_embeddings, axis=1) / sequences_lengths
    
    return embeddings


# Process sequences and get mean embeddings
def batch_compiled_sequences(token_ids_array, parameters, forward_fn, tokenizer, batch_size, embeddings_layer_to_save):
    all_embeddings = []
    # all_mean_embeddings = []
    
    for i in range(0, len(token_ids_array), batch_size):
        print(i,  'Length Token IDs Array: ', len(token_ids_array))
        batch = token_ids_array[i : i + batch_size]
        # mean_embeddings, embeddings = compiled_process_sequences(batch)
        embeddings = process_sequences(parameters, forward_fn, batch, tokenizer, embeddings_layer_to_save)
        all_embeddings.append(embeddings)        
        print(i, len(all_embeddings))

        # all_mean_embeddings.append(mean_embeddings)
        
    return all_embeddings

def main(args, num_batches, parameters, forward_fn, tokenizer, config):
    # Gather the directory and individual filenames (dir not needed here)
    raw_embed_filename, _ = create_ds_embeds_paths(input_filename=args.input_filename, args=args)

    try:
        sequences_df = pd.read_csv(args.input_filename, nrows=args.batch_size, skiprows=args.starting_row, delimiter="\t", header=None)
    except pd.errors.ParserError as e:
        print(e)


    # Get first column of sequence_df, which should contain the sequences
    sequences = sequences_df.iloc[:, 0].tolist()
    
    assert len(sequences) == args.batch_size, f"Expected {args.batch_size} sequences, but got {len(sequences)}"

    # Get token IDs from sequences using the tokenizer
    token_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
    print('Token IDs Created')
    # Convert token IDs to a JAX array
    token_ids_array = jnp.array(token_ids, dtype=jnp.int32)
        
    # mean_embeddings, embeddings = compiled_process_sequences(token_ids_array)
    embeddings = batch_compiled_sequences(token_ids_array, parameters, forward_fn, tokenizer, batch_size=100, embeddings_layer_to_save=args.embeddings_layers_to_save)

    jax.numpy.save(raw_embed_filename, embeddings, allow_pickle=True, fix_imports=True)
    print(f'\nEmbedding file index {args.starting_row} saved to {raw_embed_filename}!\n')
    # sys.exit(0)



if __name__ == "__main__":
    pass