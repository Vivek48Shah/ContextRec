

import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizerFast
from datasets import load_from_disk

# === Config ===
MODEL_DIR = "./checkpoints/contextrec_model"
DATASET_DIR = "./processed_dataset"  # path where tokenized_dataset was saved
TOP_K = 5
MAX_EXAMPLES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model and tokenizer ===
print("Loading model and tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForMaskedLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()


print("Loading tokenized dataset...")
tokenized_dataset = load_from_disk(DATASET_DIR)
test_dataset = tokenized_dataset["test"]

# === Evaluation Function ===
def evaluate_recall_at_k(dataset, k=5, max_examples=1000):
    total = 0
    hit = 0

    for example in tqdm(dataset.select(range(min(max_examples, len(dataset))))):
        input_ids = torch.tensor(example["input_ids"])
        non_pad_positions = [i for i, tid in enumerate(input_ids) if tid != tokenizer.pad_token_id]
        if len(non_pad_positions) < 2:
            continue

        mask_pos = non_pad_positions[len(non_pad_positions) // 2]
        true_token = input_ids[mask_pos].item()

        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_pos] = tokenizer.mask_token_id
        masked_input_ids = masked_input_ids.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(masked_input_ids).logits[0, mask_pos]
            top_k_preds = torch.topk(logits, k).indices.tolist()

        if true_token in top_k_preds:
            hit += 1
        total += 1

    recall_at_k = hit / total if total > 0 else 0
    print(f"Recall@{k}: {recall_at_k:.4f}")


if __name__ == "__main__":
    evaluate_recall_at_k(test_dataset, k=TOP_K, max_examples=MAX_EXAMPLES)
