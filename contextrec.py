import os
import torch
import pandas as pd
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    BertConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from collections import Counter
from tqdm import tqdm


# ========== Configuration ==========
class Config:
    RAW_DATA_PATH = "./data/events.csv"
    VOCAB_PATH = "./tokenizer/custom_vocab.txt"
    TOKENIZER_PATH = "./tokenizer"
    MODEL_DIR = "./checkpoints/contextrec_model"
    MAX_SEQ_LENGTH = 12
    MIN_SEQ_LENGTH = 3
    MASK_PROBABILITY = 0.15
    BATCH_SIZE = 16
    MAX_STEPS = 5000
    SAVE_STEPS = 1000
    LOGGING_STEPS = 100
    SEED = 42


def build_vocab(events_df):
    PAD_TOKEN = 0
    UNK_TOKEN = 1
    unique_items = sorted(events_df["itemid"].astype(str).unique())

    item2idx = {
        "[PAD]": PAD_TOKEN,
        "[UNK]": UNK_TOKEN
    }

    idx_offset = 2
    for i, item_id in enumerate(unique_items, start=idx_offset):
        item2idx[str(item_id)] = i

    os.makedirs(os.path.dirname(Config.VOCAB_PATH), exist_ok=True)
    with open(Config.VOCAB_PATH, "w") as f:
        for token in item2idx:
            f.write(token + "\n")

    return item2idx


def create_tokenizer():
    tokenizer = BertTokenizerFast(vocab_file=Config.VOCAB_PATH, do_lower_case=False)
    tokenizer.save_pretrained(Config.TOKENIZER_PATH)
    return tokenizer


def prepare_sequences(events_df):
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], unit="ms")
    events_df["date"] = events_df["timestamp"].dt.date
    sequences_df = (
        events_df.groupby(["visitorid", "date"]) ["itemid"].agg(list).reset_index()
    )
    sequences_df = sequences_df[sequences_df["itemid"].apply(len) >= Config.MIN_SEQ_LENGTH]
    sequences_df["itemid_text"] = sequences_df["itemid"].apply(lambda x: " ".join(map(str, x)))
    return sequences_df


def tokenize_dataset(sequences_df, tokenizer):
    hf_dataset = Dataset.from_pandas(sequences_df[["itemid_text"]])
    split_dataset = hf_dataset.train_test_split(test_size=0.1, seed=Config.SEED)

    def tokenize_function(example):
        return tokenizer(
            example["itemid_text"], padding="max_length", truncation=True, max_length=Config.MAX_SEQ_LENGTH
        )

    tokenized = split_dataset.map(tokenize_function, batched=True)
    return tokenized


def build_model(vocab_size):
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=Config.MAX_SEQ_LENGTH,
        type_vocab_size=1
    )
    model = BertForMaskedLM(config)
    return model


def train_model(model, tokenizer, tokenized_dataset):
    training_args = TrainingArguments(
        output_dir=Config.MODEL_DIR,
        max_steps=Config.MAX_STEPS,
        save_steps=Config.SAVE_STEPS,
        evaluation_strategy="steps",
        eval_steps=Config.SAVE_STEPS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        logging_steps=Config.LOGGING_STEPS,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.MASK_PROBABILITY
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(Config.MODEL_DIR)
    tokenizer.save_pretrained(Config.MODEL_DIR)


def main():
    print("Loading data...")
    events_df = pd.read_csv(Config.RAW_DATA_PATH)

    print("Building vocabulary and tokenizer...")
    item2idx = build_vocab(events_df)
    tokenizer = create_tokenizer()

    print("Preparing sequences...")
    sequences_df = prepare_sequences(events_df)

    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(sequences_df, tokenizer)
    tokenized_dataset.save_to_disk("./processed_dataset")

    print("Building model...")
    model = build_model(vocab_size=tokenizer.vocab_size)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Training model...")
    train_model(model, tokenizer, tokenized_dataset)

    print("Training complete. Model saved to:", Config.MODEL_DIR)


if __name__ == "__main__":
    main()
