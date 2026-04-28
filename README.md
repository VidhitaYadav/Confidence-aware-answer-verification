# Confidence-Aware Answer Verification System
A transformer-based deep learning system that verifies whether a given answer is correct for a question and context, and assigns a calibrated confidence score with explainable insights.

##  Overview
This project focuses on building a **reliability-aware AI system** that evaluates answers generated in Question Answering (QA) tasks.

Unlike traditional models, this system not only predicts correctness but also:
- Provides a **confidence score**
- Performs **confidence calibration**
- Offers **explainability**
- Analyzes **model reliability and errors**

# Confidence-Aware Answer Verification System

A lightweight NLP system that verifies whether AI-generated or user-provided answers are supported by a given context. The project detects hallucinations, assigns confidence scores, and ranks multiple answers by reliability.

## Live Demo
Hugging Face Space: https://huggingface.co/spaces/Vidhita/confidence-answer-verifier

## Features
- Auto answer generation + verification
- Manual answer verification
- AI hallucination detection
- Quora-style multi-answer ranking
- Confidence score visualization

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- Gradio
- Matplotlib

## How It Works
1. User enters a question and context
2. System generates or accepts an answer
3. Verifier checks semantic and lexical support
4. Final confidence score is computed
5. Answer is labeled as supported or likely hallucinated

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```
## Results
Here is a sample output screenshot:

![Auto Verifier](assets/Auto%20Verifier.png)
![Manual Verifier](assets/Manual%20Verifier.png)
![AI Hallucination Detector](assets/AI%20Hallucination%20Detector.png)
![Quora Style Answer Ranking](assets/Quora%20Style%20Answer%20Ranking.png)


## Future Work
- Implement retrieval-augmented verification
- Improve calibration techniques
- Explore domain-specific verification
- Add multilingual support

