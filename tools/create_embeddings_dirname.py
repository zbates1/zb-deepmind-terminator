import os

def create_ds_embeds_paths(input_filename, args):
    
    ds_embeddings_basedir = './data/ds_embeddings/'
    single_ds_embed_path = f'{ds_embeddings_basedir}{os.path.basename(input_filename).split(".")[0]}/'
    raw_embed_filename = ''
    if args is not None:
        raw_embed_filename = f'{single_ds_embed_path}/raw_embeds/start={args.starting_row}_batchsize={args.batch_size}_{args.output_basename}'
        # Make folder if it doesn't exist
        os.makedirs(os.path.dirname(raw_embed_filename), exist_ok=True)
        
    os.makedirs(single_ds_embed_path, exist_ok=True)
    
    return raw_embed_filename, single_ds_embed_path
