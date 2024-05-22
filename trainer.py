import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load AG News dataset
    logger.info("Loading AG News dataset")
    dataset = load_dataset("ag_news")
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Load the tokenizer and model
    logger.info("Loading tokenizer and model")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)

    # Preprocess the data
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    logger.info("Tokenizing datasets")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
    tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='steps',
        save_strategy='steps',
        eval_steps=500,
        save_steps=5000,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir='./logs',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )

    # Define a function to compute desired metrics
    def compute_metrics(p):
        metric = load_metric("accuracy")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        accuracy = metric.compute(predictions=predictions, references=labels)
        return accuracy

    # Initialize Trainer
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    logger.info("Starting training")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise

    # Evaluate the model
    logger.info("Evaluating the model")
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")

    # Save the trained model and tokenizer
    logger.info("Saving the model and tokenizer")
    model.save_pretrained('./results/final_model')
    tokenizer.save_pretrained('./results/final_model')

    logger.info("Script finished successfully")

if __name__ == "__main__":
    main()
