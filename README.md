# RNNs-LSTM-and-Transformer
RNNs, LSTM and Transformer

Foundational AI Project 2 – Language Modeling with RNNs, LSTMs, and Transformer

This project trains three different language models for text generation using PyTorch. The script implements an end-to-end pipeline that:

Tokenizes Input Data: Uses a BPE tokenizer (SentencePiece) to create subword tokens.
Prepares Data: Builds fixed-length token sequences from JSON Lines files.
Trains Models: Implements and trains RNN, LSTM, and Transformer models with early stopping and learning rate scheduling.
Evaluates Models: Computes evaluation metrics such as perplexity and token accuracy. It also computes BLEU scores for generated text.
Plots Loss Curves: Visualizes training and validation losses over epochs.
Generates Sample Text: Generates text using a prompt to demonstrate the models’ performance.
Saves Models: Stores the trained model weights for later use.
Measures Time: Displays the duration of each epoch and the overall training process for each model.

The project uses a custom data pipeline to:

- Read and preprocess raw training and testing data.
- Tokenize the text using a SentencePiece BPE model.
- Build sequences of fixed length.
- Train the different models and track their performance.
- Plot loss curves and generate sample outputs.

## Dependencies

To run the project, ensure you have the following installed:

- **Python 3.6+**
- **PyTorch** (tested with version ≥1.7)
- **SentencePiece** (`pip install sentencepiece`)
- **NLTK** (`pip install nltk`)
- **Matplotlib** (`pip install matplotlib`)
- **NumPy** (`pip install numpy`)

It is recommended to use a virtual environment to manage dependencies.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Molojo/RNNs-LSTM-and-Transformer.git
   cd RNNs-LSTM-and-Transformer


RNN and LSTMs/
├── .git/                          # Git version control folder
├── README.md                      # Project documentation (this file)
├── FoundationalAI_Project2.pdf    # Project report or proposal (optional)
├── train.jsonl                    # Training data file (JSON Lines format)
├── test.jsonl                     # Testing data file (JSON Lines format)
├── tokenizer.model                # Trained SentencePiece model file
├── tokenizer.vocab                # Corresponding SentencePiece vocabulary file
└── RNNs and LSTMs .py                 # Main Python script for training and evaluation

The models are evaluated using:

Cross-Entropy Loss: Computed during training.

Perplexity: Derived from the cross-entropy loss.

Token Accuracy: The percentage of correctly predicted tokens.

BLEU Score (Optional): For evaluating sample text generation quality


License
This project is licensed under the MIT License. Contributions, issues, and feature requests are welcome.

