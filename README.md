# 🧠 Fine-Tuning BERT with QLoRA on SST-2

This repository demonstrates **low-GPU-memory fine-tuning** of a BERT model using **QLoRA** on the **SST-2 Sentiment Classification** dataset.  
It shows how large transformer models can be trained efficiently on consumer GPUs (8–12 GB VRAM).

---

## 🚀 Overview

**Goal:** Efficiently fine-tune BERT using QLoRA in low-memory settings while maintaining strong performance.  
**Dataset:** SST-2 (Stanford Sentiment Treebank)  
**Frameworks:** HuggingFace Transformers, PEFT, BitsAndBytes

---

## 💻 GPU Insights from This Project

- Fine-tuning performed in a low-GPU memory setting using **4-bit QLoRA** (via bitsandbytes).  
- Achieved training on **8–12 GB VRAM GPUs** without Out-Of-Memory (OOM) errors.  
- Careful GPU memory management was critical — incorrect bitsandbytes setup or LoRA parameters caused CUDA OOM issues.  
- Efficient GPU usage enabled **faster iterations** and **multiple experiments** without high-end GPUs.

---

## 🧩 Challenges I Faced

- Configuring bitsandbytes for 4-bit QLoRA quantization.  
- Debugging CUDA memory errors.  
- Balancing learning rate and LoRA rank for stable convergence.

---

## 📈 Results

| Metric | Value |
|:-------|:------|
| **Accuracy** | ~72.9% |
| **F1-Score** | ~76.9% |
| **Loss Trend** | Steady decrease, showing good learning |

---

## 🧠 What I Learned

- QLoRA enables **low-RAM fine-tuning** of large models.
- Debugging and correct setup matter as much as model choice.
- Training efficiency can be achieved without top-tier GPUs.

---

## 🔮 Future Work

- Extend this setup to other NLP tasks.
- Experiment with larger models and optimized LoRA parameters.
- Explore deployment for **on-device inference** (Edge AI).

---

## ⚙️ Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/<your-username>/bert-qlora-sst2.git
cd bert-qlora-sst2
