# MoEEdit: Efficient and Routing-Stable Knowledge Editing for Mixture-of-Experts LLMs

This repository provides the official implementation of MoEEdit. This project introduces a method for efficient and routing-stable knowledge editing specifically designed for Mixture-of-Experts (MoE) Large Language Models.

## Requirements

**GPU**: One GPU with at least 96GB memory.



### Environment Setup

To set up the environment, we recommend using Conda to manage dependencies.

```bash
# Create a new conda environment
conda create -n moeedit python=3.12

# Activate the environment
conda activate moeedit

# Install required packages
pip install -r requirements.txt

```

## Quick Start

### 1. Edit Knowledge in LLMs

You can perform knowledge editing on supported MoE models using the `moe_edit.evaluate` module. Below is an example command for editing the `Qwen3-30B-A3B` model.

```bash
# cd to the root directory of this repository
python -m moe_edit.evaluate \
  --hparams ./configs/qwen3-30b-a3b.json \
  --model_path Qwen/Qwen3-30B-A3B \
  --dataset mcf \
  --limit 1000 \
  --data_dir ./data \
  --output_dir results

```

#### Arguments Explanation

* `hparams`: Path to the JSON file containing hyperparameters specific to the model.
* `model_path`: Local path to the model or the Hugging Face model ID (e.g., `Qwen/Qwen3-30B-A3B` or `openai/gpt-oss-20b`).
* `dataset`: The name of the dataset to be used for editing (e.g., `mcf`).
* `limit`: The maximum number of samples to edit (e.g., `1000`).
* `data_dir`: Path to the data directory. The script will automatically create the directory and download the dataset if it does not exist.
* `output_dir`: Directory where the editing results and logs will be saved.

### 2. Summarize Results

After the editing process is complete, you can summarize the editing metrics using the following command:

```bash
python -m moe_edit.summarize --dir_name MoEEdit

```


## Acknowledgement
Our code is based on [MEMIT](https://github.com/kmeng01/memit.git)

