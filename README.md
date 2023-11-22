# zb-deepmind-terminator

## Overview
This repository utilizes NVIDIA's Nucleotide Transformer to generate positional embeddings for a MPRA (Massively Parallel Reporter Assay) 3' sequence dataset. The goal is to explore the correlation between 3' sequences and gene expression.

### Resources
- **Nucleotide Transformer**: [GitHub Repository](https://github.com/instadeepai/nucleotide-transformer/tree/main)
- **Dataset Source Paper**: [PLOS Genetics Article](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1005147)

> **Note**: The Dockerfile is currently in progress and not ready for production use.

## Setup Instructions

To set up a quick conda environment for this project, follow these steps:

**1. Clone the Repository:**
```bash
git clone https://github.com/zbates1/zb-deepmind-terminator.git ./zb-deepmind-terminator && cd ./zb-deepmind-terminator
```

**2. Create the conda env**
```bash
conda create -p ./envs/zb-terminator python=3.9
```

**3. Activate Env** 
```bash
conda activate ./envs/zb-terminator
```

**4. Install Scipy with Conda** 
```bash
conda install scipy
```

**5. Clone the Nucleotide Transformer Repo**
```bash
git clone https://github.com/instadeepai/nucleotide-transformer.git ./nucleotide-transformer
```

**6. Use the native Nucleotide-Transformer setup.py file to install dependencies**
```bash
python3 ./nucleotide-transformer/setup.py install
```

**7. Then install the Nucleotide-Transformer Package=0.0.1**
```bash
cd nucleotide-transformer && pip install . && cd ..
```

**8. Install dependencies** 
```bash
pip install -r requirements.txt
```

**9. Install CUDA-enabled JAX**
```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


## Dockerfile

The Dockerfiles are available for both the PyTorch and JAX versions of this model. 


**Now you can use my pipeline to generate embeddings for the Shalem dataset (provided) and do the rest of the analysis.** 

