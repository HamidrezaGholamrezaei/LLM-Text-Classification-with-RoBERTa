# LLM-Text-Classification-with-RoBERTa

## Description
This project serves as a demonstration of advanced text classification using Large Language Models (LLMs). Leveraging the power of the RoBERTa model, it classifies text data into distinct categories with high accuracy. 
The AG News dataset is utilized as an exemplary dataset to illustrate the model's capabilities. The focus is on the effective application of the Hugging Face Transformers library to preprocess data, train a robust model, and evaluate its performance.

## Overview
- **Model**: RoBERTa (Robustly optimized BERT approach)
- **Task**: Text Classification
- **Dataset**: AG News (with four categories: World, Sports, Business, and Sci/Tech)
- **Framework**: Hugging Face Transformers
- **Language**: Python

## Usage
Set up the project environment on your machine by following these steps:

1. Clone the repository:
    ```
    git clone https://github.com/HamidrezaGholamrezaei/LLM-Text-Classification-with-RoBERTa.git
    cd LLM-Text-Classification-with-RoBERTa
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Run the Training Script:
    ```
    python train.py
    ```

## Results
After training, the model achieved the following evaluation metrics:
- **Accuracy**: 95.3%
- **Evaluation Loss**: 0.265
- **Train Loss**: 0.168

The trained model and tokenizer are saved in the ./results/final_model directory for future use.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. For more detailed information, please refer to the LICENSE file.