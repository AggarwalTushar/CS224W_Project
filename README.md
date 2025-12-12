# CS224W_Project
Project work for CSS224W Course (Autumn 25)

## Project Members
- Tushar Aggarwal (email: tushar53@stanford.edu)
- Brandon Liu (email: liubr@stanford.edu)
- Mete Gumusayak (email: mete1@stanford.edu)

## Project Overview
This project focuses on predicting future earthquake events Graph Neural Networks (GNNs). We utilize a real-world dataset, where nodes represent geographic locations and times and edges represent interactions. The goal is to predict whether a location will face earthquake in the future based on historical data.

## Repository Structure
- `data/`: Contains the dataset and the synthetic data generation scripts.
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
3. Run the training script:
   To train the GraphSage model, execute the following command:
   ```bash
   python src/train_graphsage.py
   ```
   For all the other models, we have provided a Jupyter notebook `train_{model_name}.ipynb` in the `src/` directory. Open the notebook and run the cells sequentially to train and evaluate the models.


## Acknowledgments
We would like to thank the instructors, our mentor TA (Harper Hua) and all other TAs of the CS224W course for their guidance and support throughout the quarter. We are also deeply thankful to Prof. William Ellsworth for his valuable advice and insightful discussions, which greatly strengthened this work.