# Transformer-Based Text Classification with Flash Attention

A PyTorch implementation of an efficient transformer-based text classification pipeline, leveraging Flash Attention for improved performance and scalability.


## Overview

This project implements a transformer architecture adapted for **text classification tasks**, inspired by decoder-style models and optimized with **Flash Attention** for efficient computation.

It includes:
* Custom tokenization pipeline
* Transformer-based classification model
* Efficient attention mechanism (Flash Attention)
* End-to-end training and evaluation pipeline


## Features

* **Flash Attention integration** for faster and memory-efficient training
* **Transformer-based architecture** adapted for classification
* **Custom tokenizer support** for preprocessing text data
* Training and evaluation pipeline with standard classification metrics
* Modular structure for experimentation and extension


## Project Structure

```
.
├── base.py                  # Base model utilities
├── basic.py                 # Baseline implementations
├── model.py                 # Transformer model definition
├── transformer_approach.py  # Transformer + Flash Attention implementation
├── pipeline.py              # Training and evaluation pipeline
├── train_refinement.py      # Training improvements and tuning
├── inspect_data.py          # Data analysis utilities
├── baseline.ipynb           # Baseline experiments
├── playground.ipynb         # Experimentation notebook
```


### Train the model

```bash
python pipeline.py
```

### Run experiments / analysis

```bash
python inspect_data.py
```

## Key Idea

Unlike standard transformer implementations focused on text generation, this project adapts a **decoder-style transformer architecture for classification tasks**, while incorporating **Flash Attention** to improve computational efficiency.

