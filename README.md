# LSTM-Based Next-Word Prediction Text

This project implements a next-word prediction model for Thai text using Long Short-Term Memory (LSTM) networks and natural language processing techniques. The model is trained to predict the next word given a sequence of words from Thai text.

## Overview

The project includes a Python script `model.py` that performs the following tasks:

1. **Data Preparation**: Reads Thai text data, tokenizes it, and prepares sequences for training.
2. **Model Training**: Builds and trains an LSTM-based model for predicting the next word in a sequence.
3. **Prediction**: Uses the trained model to predict the next word based on a given input text.

## Project Structure

- `model.py`: The main script that contains the code for data preparation, model training, and prediction.
- `data/language.txt`: The text file containing the Thai language data used for training.

## Installation

Ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install numpy tensorflow pythainlp
```

## Usage

1. **Prepare Data**: Ensure that `data/language.txt` contains the Thai text data you want to use for training.

2. **Run the Script**: Execute the `model.py` script to train the model and make predictions:

   ```bash
   python model.py
   ```

3. **Input Text Prediction**: The script will print the predicted next word for the input text "วันนี้".

## Code Details

### Data Preparation

The script reads the text from `data/language.txt`, tokenizes it using `pythainlp`, and creates sequences of a specified length to be used for training.

### Model

- **Embedding Layer**: Converts words into dense vectors of fixed size.
- **LSTM Layer**: A Long Short-Term Memory layer to capture dependencies in sequences.
- **Dense Layer**: A fully connected layer with a softmax activation function to output probabilities for each word in the vocabulary.

### Training

The model is compiled with categorical crossentropy loss and the Adam optimizer. It is trained for 10 epochs.

### Prediction

The trained model predicts the next word based on an input text. The predicted word is displayed in the console.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [PyThaiNLP](https://pythainlp.github.io/) for Thai language tokenization.
- [TensorFlow](https://www.tensorflow.org/) for the machine learning framework.
