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
git clone https://github.com/zbates1/zb-deepmind-terminator.git ./deepmind-terminator && cd ./deepmind-terminator
```

**2. Create the conda env**
```bash
conda create -n ./envs/zb-terminator python=3.8
```

**3. Activate Env** 
```bash
conda activate zb-terminator
```

**4. Install dependencies** 
```bash
pip install -r requirements.txt
```

**5. Clone the Nucleotide Transformer Repo**
```bash
git clone https://github.com/instadeepai/nucleotide-transformer.git ./nt-terminator && cd ./nt-terminator
```

**6. Install Nucleotide Transformer**
```bash
pip install .
```

**Now you can use my pipeline to generate embeddings for the Shalem dataset (provided) and do the rest of the analysis.** 