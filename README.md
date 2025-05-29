# Text to Trust: Evaluating Language Model Trade-offs for Unfair Terms of Service Detection

This repository presents our work on automated detection of potentially unfair clauses in Terms of Service (ToS) documents using a combination of **fine-tuned**, **parameter-efficient**, and **zero-shot large language models (LLMs)**.

## 🚀 Project Overview

Modern digital contracts often include clauses that are hard to interpret and potentially unfair to users. We aim to build a scalable and cost-effective system for identifying such clauses using state-of-the-art NLP techniques.

We experimented with multiple approaches:

- **Full fine-tuning** of BERT and DistilBERT  
- **LoRA-based PEFT** on TinyLlama, LLaMA, and SaulLM using 4-bit quantization  
- **Zero-shot prompting** using high-performing models like GPT-4o and O3-mini

Our pipeline was evaluated on the **Claudette-ToS** benchmark and validated on the real-world **Multilingual Scraper of Privacy Policies and Terms of Service** corpus.

📝 **[Read the paper here](https://github.com/sravanigona/UnfairTOSAgreementsDetection/blob/main/unfair-tos-detection-nlp-paper-umass.pdf)**

---

## 📊 Results Summary

| Model          | Type    | F1 Score | Recall | Precision |
|----------------|---------|----------|--------|-----------|
| BERT           | Full FT | 81.6     | 82.5   | 80.7      |
| SaulLM-7B      | LoRA    | 78.5     | 97.5   | 66.0      |
| TinyLlama-1.1B | LoRA    | 66.1     | 52.0   | 89.2      |
| GPT-4o         | Zero-Shot | 72.3   | 89.1   | 61.0      |

---

## 🧠 Key Takeaways

- **Full fine-tuning (BERT)** yields the best overall performance.
- **LoRA models** provide lightweight alternatives with strong precision or recall.
- **Zero-shot prompting** is useful for fast deployment but prone to false positives.
- The best model was deployed on a real-world ToS dataset, successfully identifying over 150 suspicious clauses.

---

## 📄 Datasets Used

- [Claudette-ToS (Hugging Face)](https://huggingface.co/datasets/LawInformedAI/claudette_tos)
- [Multilingual Scraper Corpus](https://karelkubicek.github.io/assets/pdf/Multilingual_Scraper_of_Privacy_Policies_and_Terms_of_Service.pdf)

---

## 👥 Contributors

- **Sravani Gona**  
- **Sahithi Singireddy**  
- **Noshitha Padma Pratyusha Juttu**  
- **Sujal Timilsina**
