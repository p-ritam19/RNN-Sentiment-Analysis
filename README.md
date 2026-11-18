# RNN for Sentiment Analysis

A sentiment analysis project using the IMDB movie reviews dataset, text preprocessing, TF-IDF vectorization, and a Recurrent Neural Network (RNN) implemented with PyTorch.

## Project Overview

This project classifies movie reviews into two sentiment categories:

- Positive
- Negative

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- PyTorch
- Jupyter Notebook

## Dataset

The project uses the IMDB Dataset containing movie reviews and their sentiment labels.

Dataset file: `IMDB Dataset.csv`

## Workflow

1. Load and explore the dataset.
2. Remove duplicate reviews.
3. Convert text to lowercase.
4. Remove URLs, punctuation, and HTML tags.
5. Remove English stopwords.
6. Apply Porter stemming.
7. Encode sentiment labels.
8. Convert reviews into TF-IDF features using a maximum of 5,000 features.
9. Split the dataset into training and testing sets (80:20).
10. Build and train a PyTorch RNN.
11. Evaluate the model using test accuracy.

## Model Architecture

- RNN hidden size: 128
- Number of RNN layers: 1
- Fully connected output layer: 1 neuron
- Activation: Sigmoid
- Loss function: Binary Cross Entropy
- Optimizer: Adam
- Epochs: 10
- Batch size: 64

Note: The current implementation feeds each TF-IDF feature vector to the RNN as a single time step.

## Project Structure

```text
RNN for Sentiment Analysis/
│
├── IMDB Dataset.csv
├── RNN_for_sentimentanalysis.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the required NLTK resources:

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

## Run the Project

Open `RNN_for_sentimentanalysis.ipynb` in Jupyter Notebook and execute the cells in order.

## Future Improvements

- Experiment with word-level sequence representations.
- Explore LSTM and GRU architectures.
- Improve model evaluation with precision, recall, and F1-score.
- Experiment with different preprocessing and vectorization techniques.
