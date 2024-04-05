from typing import Dict

import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import DatasetDict, Dataset, load_dataset


def load_and_tokenize_data(tokenizer: BertTokenizer) -> DatasetDict:
    dataset: DatasetDict = load_dataset('glue', 'mrpc')

    def tokenize(batch: Dict[str, str]) -> Dict[str, Dict[str, int]]:
        return tokenizer(batch['sentence1'], batch['sentence2'], 
                         padding='max_length', truncation=True, max_length=512)

    dataset: DatasetDict = dataset.map(
        tokenize, batched=True, batch_size=len(dataset))

    dataset.set_format('torch', columns=[
                       'input_ids', 'attention_mask', 'label'])

    return dataset


def create_trainer(model: BertForSequenceClassification,
                   train_dataset: Dataset,
                   eval_dataset: Dataset) -> Trainer:
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    return trainer


def main():
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased')
    model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased')

    dataset: DatasetDict = load_and_tokenize_data(tokenizer)
    train_dataset, eval_dataset = dataset['train'], dataset['validation']

    trainer: Trainer = create_trainer(model, train_dataset, eval_dataset)

    trainer.train()

    print(f'Peak memory usage after training: {torch.cuda.max_memory_allocated() / 1e9:.4f} GB')

    trainer.save_model()
    trainer.evaluate()

    print(f'Peak memory usage after evaluation: {torch.cuda.max_memory_allocated() / 1e9:.4f} GB')


if __name__ == "__main__":
    main()
