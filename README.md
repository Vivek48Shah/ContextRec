# ContextRec


# Contextual Sequence-Based Retrieval System using Custom BERT MLM

**ContextRec** is a contextual sequence-based retrieval system built on a custom-trained BERT-style Masked Language Model (MLM). It models user-item interaction sequences to learn contextual relationships between items, enabling personalized and intelligent retrieval and recommendations.

Unlike traditional recommender systems that rely on static co-occurrence or collaborative filtering, ContextRec uses deep sequence modeling to understand behavior in context—much like how language models interpret word sequences.

---

## Overview

ContextRec treats item interaction sequences as "sentences" and learns to predict masked items based on their surrounding context. This approach captures both short-term and long-term dependencies in user behavior, making it well-suited for personalized retrieval tasks.

---

## What This System Does

- Learns contextual item embeddings from sequential interaction data
- Models session-level behavior using transformer-based MLM
- Predicts masked items, enabling intelligent item retrieval
- Supports real-time recommendations by understanding a user's current session

---

## Why Context-Based Sequence Modeling?

### Contextual Awareness
Item meaning is highly contextual. The same item may mean different things depending on what comes before or after it in a session.

### Temporal & Sequential Modeling
Users follow behavioral patterns. Modeling those sequences allows the system to predict and retrieve items based on session progression.

### Generalization & Flexibility
ContextRec doesn't just memorize co-occurrence—it learns item relationships in context, making it more generalizable across domains.

---

## How It Works

- **Custom Vocabulary**: Built from unique product/item IDs with required special tokens (`[PAD]`, `[MASK]`, `[SEP]`, etc.).
- **Tokenizer**: `BertTokenizerFast` tokenizes sequences of item IDs.
- **Model**: `BertForMaskedLM` is trained from scratch with a config that matches the custom vocab.
- **Objective**: Masked Language Modeling (MLM), where items in a sequence are randomly masked and predicted based on context.

---

## Problem Formulation

ContextRec is formulated as a **Masked Item Prediction** task, inspired by Masked Language Modeling (MLM). Each user session is represented as a sequence of item IDs:
[item_1021, item_2099, [MASK], item_4762, item_6780]

The model learns to predict the masked item based on its surrounding context using a BERT-style transformer architecture.

---

## Evaluation

We evaluate the model using **Recall@K**, a common metric in recommendation and retrieval:

- For each test sequence, we mask one item.
- The model predicts the top-K most likely items for the masked position.
- If the true item is within those K predictions, it's considered a **hit**.

### Metrics:

- **Recall@K**:  
  Proportion of masked positions where the true item appears in the top-K predictions.


| Metric        | Value     |
|---------------|-----------|
| Recall@1      | 0.28     |
| Recall@5      | 0.34     |
| Recall@20     | 0.65     |
---
  
## Applications

ContextRec can be integrated into various stages of the user journey:

- **Search Re-ranking**: Reorder search results based on session context
- **Item Detail Page (IDP)**: Suggest complementary or similar items while viewing a product
- **Cart Recommendations**: Show relevant add-ons or bundles
- **Pre-Checkout**: Surface high-converting cross-sell items
- **Post-Checkout**: Recommend next-best actions or repeat purchases

---

## Extensibility

The current implementation uses **only product IDs** to build sequences and train the model. However, this architecture is extensible. We can enhance it by incorporating:

- Product category or brand embeddings
- User demographics or behavior segments
- Temporal features (e.g., time of day, day of week)
- Event types (view, cart, purchase)

By including these contextual signals, the model can learn richer relationships and deliver even more accurate and personalized recommendations.


