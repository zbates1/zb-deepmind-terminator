# load_model.py

import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_model
import nucleotide_transformer

def load_nt_model(model_name, embeddings_layers_to_save, max_positions):
    # Instantiate the model
    # Get pretrained model
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name=model_name,
    embeddings_layers_to_save=embeddings_layers_to_save,
    max_positions=max_positions
)
    print('Passed model loading with embeddings_layers_to_save set to: ', embeddings_layers_to_save)
    forward_fn = hk.transform(forward_fn)
    print('Passed forward_fn')
    
    return parameters, forward_fn, tokenizer, config
