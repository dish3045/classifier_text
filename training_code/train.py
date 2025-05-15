import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    pipeline
)
import torch

def load_dailydialog_dataset():
    train_df = pd.read_csv("../dataset/train.csv")
    val_df = pd.read_csv("../dataset/validation.csv")
    test_df = pd.read_csv("../dataset/test.csv")

    train_data = Dataset.from_pandas(train_df)
    val_data = Dataset.from_pandas(val_df)
    test_data = Dataset.from_pandas(test_df)

    return DatasetDict({
        'train': train_data,
        'validation': val_data,
        'test': test_data
    })

def preprocess_dataset(datasets, tokenizer):
    def preprocess_function(examples):
        dialogues = examples["dialog"] if isinstance(examples["dialog"], list) else [examples["dialog"]]
        inputs = [" ".join(eval(d)) if isinstance(d, str) else " ".join(d) for d in dialogues]
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["emotion"], padding="max_length", truncation=True, max_length=10)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return datasets.map(preprocess_function, batched=True)

def fine_tune_model(datasets, model_name="Tianlin668/MentalT5"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    datasets = preprocess_dataset(datasets, tokenizer)

    training_args = TrainingArguments(
        output_dir="../models",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=2,
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model("../models")
    tokenizer.save_pretrained("../models")
    return model, tokenizer

def predict_emotion(text, model_path="../models"):
    classifier = pipeline("text2text-generation", model=model_path, tokenizer=model_path)
    result = classifier(text)
    return result[0]['generated_text']

if __name__ == "__main__":
    print("Loading and preprocessing dataset...")
    datasets = load_dailydialog_dataset()

    print("Fine-tuning the model...")
    model, tokenizer = fine_tune_model(datasets)

    test_input = "I can't believe I got the job! I'm so excited!"
    emotion = predict_emotion(test_input)
    print(f"Predicted Emotion: {emotion}")
