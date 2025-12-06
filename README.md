# CS224W_Project
Project work for CSS224W Course (Autumn 25)

## Project Members
- Tushar Aggarwal (email: tushar53@stanford.edu)
- Brandon Li (email: liubr@stanford.edu)
- Mete Gumusayak (email: mete1@stanford.edu)

## Project Overview
This project focuses on predicting future earthquake events Graph Neural Networks (GNNs). We utilize a real-world dataset, where nodes represent geographic locations and times and edges represent interactions. The goal is to predict whether a location will face earthquake in the future based on historical data.

## Repository Structure
- `data/`: Contains the dataset.
- `src/`: Contains source code for data processing, model training, and evaluation.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `results/`: Contains results from model training and evaluation.

## Getting Started
1. Clone the repository:
   ```bash
    git clone
    https://github.com/AggarwalTushar/CS224W_Project.git
    cd CS224W_Project
    ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train the transformer model
```bash
# For default
python src/train_transformer.py 
# Custom configuration
python src/train_transformer.py --epochs 100 --batch_size 16 --hidden_dim 16 --num_layers 2 --num_heads 4 --dropout 0.5
# For help
python src/train_transformer.py --help
```

## Acknowledgments
We would like to thank the instructors and TAs of the CS224W course for their guidance and support throughout the quarter.