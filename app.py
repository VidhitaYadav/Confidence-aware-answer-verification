import textwrap

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer


APP_TITLE = "Confidence-Aware Answer Verification System"
QA_MODEL = "distilbert-base-cased-distilled-squad"
NLI_MODEL = "cross-encoder/nli-distilroberta-base"
CONFIDENCE_THRESHOLD = 0.58
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on", "at", "for",
    "and", "or", "by", "with", "from", "that", "this", "it", "as", "be", "been", "being",
}


def safe_strip(value):
    return value.strip() if isinstance(value, str) else ""


print("Loading QA generator...")
qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
print("Loading NLI verifier...")
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
print("Models loaded successfully.")


def normalize_label(label):
    return label.lower().strip()


def normalize_text(text):
    text = safe_strip(text).lower()
    cleaned = []
    for ch in text:
        cleaned.append(ch if ch.isalnum() or ch.isspace() else " ")
    return " ".join("".join(cleaned).split())


def content_tokens(text):
    return [token for token in normalize_text(text).split() if token not in STOPWORDS]


def token_f1(text_a, text_b):
    tokens_a = content_tokens(text_a)
    tokens_b = content_tokens(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    overlap = len(set_a & set_b)
    if overlap == 0:
        return 0.0
    precision = overlap / max(len(set_b), 1)
    recall = overlap / max(len(set_a), 1)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def lexical_support_score(context, answer):
    norm_context = normalize_text(context)
    norm_answer = normalize_text(answer)
    if not norm_context or not norm_answer:
        return 0.0
    if norm_answer in norm_context:
        return 1.0

    answer_tokens = content_tokens(answer)
    if not answer_tokens:
        return 0.0

    context_tokens = set(content_tokens(context))
    overlap = sum(1 for token in answer_tokens if token in context_tokens)
    return overlap / len(answer_tokens)


def build_hypothesis(question, answer):
    question = safe_strip(question)
    answer = safe_strip(answer)
    if not question or not answer:
        return ""
    return f"For the question '{question}', the correct answer is '{answer}'."


def score_with_nli(context, question, answer):
    premise = safe_strip(context)
    hypothesis = build_hypothesis(question, answer)
    inputs = nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = nli_model(**inputs)

    probs = torch.softmax(outputs.logits[0], dim=0)
    id2label = nli_model.config.id2label
    scores = {normalize_label(id2label[i]): float(probs[i].item()) for i in range(len(probs))}
    entailment = scores.get("entailment", 0.0)
    contradiction = scores.get("contradiction", 0.0)
    neutral = scores.get("neutral", 0.0)

    support = entailment + 0.5 * neutral
    oppose = contradiction + 0.15 * neutral
    confidence = support / max(support + oppose, 1e-8)

    return {
        "nli_confidence": confidence,
        "entailment": entailment,
        "contradiction": contradiction,
        "neutral": neutral,
    }


def verify_answer(question, context, answer):
    nli_result = score_with_nli(context, question, answer)
    generated = generate_answer(question, context)
    lexical_score = lexical_support_score(context, answer)
    qa_match_score = token_f1(answer, generated["answer"])

    final_confidence = (
        0.40 * nli_result["nli_confidence"]
        + 0.35 * lexical_score
        + 0.25 * qa_match_score
    )

    if lexical_score == 1.0:
        final_confidence = max(final_confidence, 0.88)
    elif qa_match_score >= 0.80:
        final_confidence = max(final_confidence, 0.82)

    label = "SUPPORTED" if final_confidence >= CONFIDENCE_THRESHOLD else "LIKELY_HALLUCINATION"
    return {
        "confidence": final_confidence,
        "entailment": nli_result["entailment"],
        "contradiction": nli_result["contradiction"],
        "neutral": nli_result["neutral"],
        "nli_confidence": nli_result["nli_confidence"],
        "lexical_score": lexical_score,
        "qa_match_score": qa_match_score,
        "reference_answer": generated["answer"],
        "reference_qa_score": generated["qa_score"],
        "label": label,
    }


def reliability_band(score):
    if score >= 0.82:
        return "High", "Strongly supported by the context."
    if score >= 0.68:
        return "Moderate", "Mostly supported, but still worth checking."
    if score >= CONFIDENCE_THRESHOLD:
        return "Borderline", "Partially supported. Treat with caution."
    if score >= 0.40:
        return "Low", "Weak support. This may be inaccurate."
    return "Very Low", "The answer is likely hallucinated or contradicted."


def generate_answer(question, context):
    question = safe_strip(question)
    context = safe_strip(context)
    inputs = qa_tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    start_index = int(torch.argmax(start_logits))
    end_index = int(torch.argmax(end_logits))

    if end_index < start_index:
        end_index = start_index

    answer_ids = inputs["input_ids"][0][start_index : end_index + 1]
    answer = qa_tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    start_prob = torch.softmax(start_logits, dim=0)[start_index].item()
    end_prob = torch.softmax(end_logits, dim=0)[end_index].item()
    qa_score = (start_prob + end_prob) / 2.0

    return {
        "answer": answer,
        "qa_score": float(qa_score),
    }


def gauge_plot(score, title):
    fig, ax = plt.subplots(figsize=(6.6, 1.8))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#111827")

    if score >= 0.82:
        color = "#22c55e"
    elif score >= CONFIDENCE_THRESHOLD:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    ax.barh([""], [score], color=color, height=0.6)
    ax.barh([""], [1 - score], left=[score], color="#1f2937", height=0.6)
    ax.axvline(CONFIDENCE_THRESHOLD, color="white", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="#d1d5db", fontsize=8)
    ax.set_title(title, color="white", fontsize=11, pad=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")
    plt.tight_layout()
    return fig


def metrics_markdown(question, context, answer, verification, qa_score=None):
    score = verification["confidence"]
    band, message = reliability_band(score)
    icon = "✅" if verification["label"] == "SUPPORTED" else "❌"
    verdict = "SUPPORTED" if verification["label"] == "SUPPORTED" else "LIKELY HALLUCINATION"

    lines = [
        f"## {icon} Verification Result: {verdict}",
        "",
        f"**Question:** {question}",
        "",
        f"**Answer being checked:** {answer}",
        "",
        f"**Reliability band:** {band}",
        "",
        message,
        "",
        "| Metric | Score |",
        "|---|---:|",
        f"| Confidence | `{score:.4f}` |",
        f"| NLI confidence | `{verification['nli_confidence']:.4f}` |",
        f"| Lexical support | `{verification['lexical_score']:.4f}` |",
        f"| QA answer match | `{verification['qa_match_score']:.4f}` |",
        f"| Entailment | `{verification['entailment']:.4f}` |",
        f"| Contradiction | `{verification['contradiction']:.4f}` |",
        f"| Neutral | `{verification['neutral']:.4f}` |",
    ]
    if qa_score is not None:
        lines.append(f"| QA extraction confidence | `{qa_score:.4f}` |")
    if verification.get("reference_answer"):
        lines.append(f"| Reference answer from QA | `{verification['reference_answer']}` |")
    lines.extend(
        [
            "",
            f"**Rule used:** final confidence combines NLI + lexical evidence + QA agreement. Score >= `{CONFIDENCE_THRESHOLD:.2f}` means supported.",
            "",
            f"**Context excerpt:** {textwrap.shorten(context, width=260, placeholder='...')}",
        ]
    )
    return "\n".join(lines)


def manual_verify(question, context, answer):
    question = safe_strip(question)
    context = safe_strip(context)
    answer = safe_strip(answer)
    if not question or not context or not answer:
        return "Please fill in question, context, and answer.", None

    verification = verify_answer(question, context, answer)
    plot = gauge_plot(verification["confidence"], f"Confidence: {verification['confidence'] * 100:.1f}%")
    return metrics_markdown(question, context, answer, verification), plot


def auto_verify(question, context):
    question = safe_strip(question)
    context = safe_strip(context)
    if not question or not context:
        return "Please fill in question and context.", "", None

    generated = generate_answer(question, context)
    if not generated["answer"]:
        return "The QA model could not extract an answer from the provided context.", "", None

    verification = verify_answer(question, context, generated["answer"])
    plot = gauge_plot(verification["confidence"], f"Confidence: {verification['confidence'] * 100:.1f}%")
    report = metrics_markdown(
        question,
        context,
        generated["answer"],
        verification,
        qa_score=generated["qa_score"],
    )
    answer_box = (
        f"### AI-Generated Answer\n\n"
        f"`{generated['answer']}`\n\n"
        f"Extraction confidence: `{generated['qa_score']:.4f}`"
    )
    return report, answer_box, plot


AI_CASES = [
    {
        "title": "History",
        "question": "Who invented the telephone?",
        "context": (
            "Alexander Graham Bell is widely credited with inventing the first practical telephone "
            "and received the first U.S. patent for the device in 1876."
        ),
        "answers": [
            {"source": "Chatbot A", "answer": "Alexander Graham Bell invented the telephone in 1876."},
            {"source": "Chatbot B", "answer": "Thomas Edison invented the telephone in 1890."},
            {"source": "Chatbot C", "answer": "The telephone is associated with Alexander Graham Bell."},
        ],
    },
    {
        "title": "Geography",
        "question": "What is the capital of Australia?",
        "context": (
            "Canberra is the capital city of Australia. Sydney is the largest city, "
            "but it is not the capital."
        ),
        "answers": [
            {"source": "Chatbot A", "answer": "Canberra is the capital of Australia."},
            {"source": "Chatbot B", "answer": "Sydney is the capital of Australia."},
            {"source": "Chatbot C", "answer": "Australia's capital is Canberra, not Sydney."},
        ],
    },
    {
        "title": "Science",
        "question": "What is the boiling point of water at standard atmospheric pressure?",
        "context": (
            "At standard atmospheric pressure, pure water boils at 100 degrees Celsius, "
            "which is equal to 212 degrees Fahrenheit."
        ),
        "answers": [
            {"source": "Chatbot A", "answer": "Water boils at 100 degrees Celsius."},
            {"source": "Chatbot B", "answer": "Water boils at 80 degrees Celsius."},
            {"source": "Chatbot C", "answer": "At sea level, the boiling point is 212 degrees Fahrenheit."},
        ],
    },
]


def hallucination_demo(case_index):
    case = AI_CASES[int(case_index)]
    chart_labels = []
    chart_scores = []
    chart_colors = []

    lines = [
        f"## AI Hallucination Detector: {case['title']}",
        "",
        f"**Question:** {case['question']}",
        "",
        f"**Reference context:** {case['context']}",
        "",
    ]

    for item in case["answers"]:
        verification = verify_answer(case["question"], case["context"], item["answer"])
        supported = verification["label"] == "SUPPORTED"
        badge = "✅ Supported" if supported else "❌ Hallucination caught"
        lines.extend(
            [
                f"### {item['source']} - {badge}",
                f"**Answer:** {item['answer']}",
                f"Confidence: `{verification['confidence']:.3f}` | Entailment: `{verification['entailment']:.3f}` | Contradiction: `{verification['contradiction']:.3f}`",
                "",
            ]
        )
        chart_labels.append(item["source"])
        chart_scores.append(verification["confidence"])
        chart_colors.append("#22c55e" if supported else "#ef4444")

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#111827")
    bars = ax.barh(chart_labels, chart_scores, color=chart_colors, height=0.55)
    ax.axvline(CONFIDENCE_THRESHOLD, color="white", linewidth=1.6, linestyle="--", alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Verification confidence", color="white")
    ax.set_title("Green = supported, Red = hallucination", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")
    for bar, score in zip(bars, chart_scores):
        ax.text(min(score + 0.02, 0.92), bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center", color="white")
    plt.tight_layout()

    return "\n".join(lines), fig


RANKING_CASES = [
    {
        "title": "Quora-style ranking",
        "question": "What planet is closest to the Sun?",
        "context": (
            "Mercury is the closest planet to the Sun in our solar system. "
            "Venus is the second planet from the Sun."
        ),
        "answers": [
            "Mercury is the closest planet to the Sun.",
            "Venus is the closest planet to the Sun.",
            "Mercury is nearest to the Sun in our solar system.",
            "Earth is the closest planet to the Sun.",
        ],
    },
    {
        "title": "Student answer ranking",
        "question": "Who painted the Mona Lisa?",
        "context": (
            "The Mona Lisa was painted by Leonardo da Vinci during the Renaissance. "
            "It is one of the most famous paintings in the world."
        ),
        "answers": [
            "Leonardo da Vinci painted the Mona Lisa.",
            "Vincent van Gogh painted the Mona Lisa.",
            "The Mona Lisa is a work by Leonardo da Vinci.",
            "Pablo Picasso painted the Mona Lisa.",
        ],
    },
]


def rank_answers(question, context, answers_blob):
    question = safe_strip(question)
    context = safe_strip(context)
    answers = [line.strip() for line in answers_blob.splitlines() if line.strip()]
    if not question or not context or not answers:
        return "Please provide question, context, and at least one answer.", None

    ranked = []
    for answer in answers:
        verification = verify_answer(question, context, answer)
        ranked.append((verification["confidence"], answer, verification))
    ranked.sort(key=lambda item: item[0], reverse=True)

    lines = [
        "## Ranked Answers",
        "",
        f"**Question:** {question}",
        "",
    ]
    for index, (score, answer, verification) in enumerate(ranked, start=1):
        verdict = "Reliable" if verification["label"] == "SUPPORTED" else "Unreliable"
        lines.extend(
            [
                f"### Rank {index} - {verdict}",
                f"**Answer:** {answer}",
                f"Confidence: `{score:.3f}` | Entailment: `{verification['entailment']:.3f}` | Contradiction: `{verification['contradiction']:.3f}`",
                "",
            ]
        )

    labels = [textwrap.shorten(item[1], width=34, placeholder="...") for item in ranked]
    scores = [item[0] for item in ranked]
    colors = ["#22c55e" if item[2]["label"] == "SUPPORTED" else "#ef4444" for item in ranked]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#111827")
    bars = ax.barh(labels, scores, color=colors, height=0.55)
    ax.axvline(CONFIDENCE_THRESHOLD, color="white", linewidth=1.6, linestyle="--", alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Reliability score", color="white")
    ax.set_title("Higher score = more supported by context", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")
    for bar, score in zip(bars, scores):
        ax.text(min(score + 0.02, 0.92), bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center", color="white", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    return "\n".join(lines), fig


def load_ranking_case(index):
    case = RANKING_CASES[int(index)]
    answers_blob = "\n".join(case["answers"])
    return case["question"], case["context"], answers_blob


AUTO_EXAMPLES = [
    [
        "Who invented the telephone?",
        "Alexander Graham Bell is widely credited with inventing the first practical telephone and received the first U.S. patent for the device in 1876.",
    ],
    [
        "What is the capital of Australia?",
        "Canberra is the capital city of Australia. Sydney is the largest city, but it is not the capital.",
    ],
    [
        "What is the boiling point of water?",
        "At standard atmospheric pressure, pure water boils at 100 degrees Celsius, which equals 212 degrees Fahrenheit.",
    ],
]


MANUAL_EXAMPLES = [
    [
        "What is the capital of Australia?",
        "Canberra is the capital city of Australia. Sydney is the largest city, but it is not the capital.",
        "Sydney is the capital of Australia.",
    ],
    [
        "Who invented the telephone?",
        "Alexander Graham Bell is widely credited with inventing the first practical telephone and received the first U.S. patent for the device in 1876.",
        "Alexander Graham Bell",
    ],
    [
        "Who invented the telephone?",
        "Alexander Graham Bell is widely credited with inventing the first practical telephone and received the first U.S. patent for the device in 1876.",
        "Thomas Edison",
    ],
]


def load_auto_example(index):
    example = AUTO_EXAMPLES[int(index)]
    return example[0], example[1], "### AI-Generated Answer\n\nResult will appear here.", "Verification report will appear here.", None


def load_manual_example(index):
    example = MANUAL_EXAMPLES[int(index)]
    return example[0], example[1], example[2], "Verification report will appear here.", None


CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
}
"""


with gr.Blocks(title=APP_TITLE, css=CUSTOM_CSS) as demo:
    gr.Markdown(
        f"""
# {APP_TITLE}
### Verify answers using context-aware confidence scoring
"""
    )


    with gr.Tabs():
        with gr.TabItem("Auto Verifier (Real AI Pipeline)"):
            gr.Markdown("Give a question and a context passage. The QA model generates an answer, then the verifier checks whether that answer is actually supported.")
            with gr.Row():
                with gr.Column():
                    auto_example_choice = gr.Radio(
                        choices=[("Telephone", "0"), ("Australia", "1"), ("Boiling point", "2")],
                        value="0",
                        label="Load an example",
                    )
                    auto_load_btn = gr.Button("Load Auto Example")
                    auto_q = gr.Textbox(label="Question", lines=2, placeholder="Who invented the telephone?")
                    auto_ctx = gr.Textbox(
                        label="Reference context",
                        lines=6,
                        placeholder="Alexander Graham Bell is widely credited with inventing the first practical telephone...",
                    )
                    auto_btn = gr.Button("Generate Answer + Verify", variant="primary")
                with gr.Column():
                    auto_answer = gr.Markdown("### AI-Generated Answer\n\nResult will appear here.")
                    auto_report = gr.Markdown("Verification report will appear here.")
                    auto_plot = gr.Plot()
            auto_load_btn.click(load_auto_example, [auto_example_choice], [auto_q, auto_ctx, auto_answer, auto_report, auto_plot])
            auto_btn.click(auto_verify, [auto_q, auto_ctx], [auto_report, auto_answer, auto_plot])

        with gr.TabItem("Manual Verifier"):
            gr.Markdown("Paste any answer from ChatGPT, Gemini, Quora, a student, or your own system and verify whether the context supports it.")
            with gr.Row():
                with gr.Column():
                    man_example_choice = gr.Radio(
                        choices=[("Wrong capital", "0"), ("Correct short answer", "1"), ("Wrong person", "2")],
                        value="1",
                        label="Load an example",
                    )
                    man_load_btn = gr.Button("Load Manual Example")
                    man_q = gr.Textbox(label="Question", lines=2)
                    man_ctx = gr.Textbox(label="Reference context", lines=6)
                    man_ans = gr.Textbox(label="Answer to verify", lines=2)
                    man_btn = gr.Button("Verify Answer", variant="primary")
                with gr.Column():
                    man_report = gr.Markdown("Verification report will appear here.")
                    man_plot = gr.Plot()
            man_load_btn.click(load_manual_example, [man_example_choice], [man_q, man_ctx, man_ans, man_report, man_plot])
            man_btn.click(manual_verify, [man_q, man_ctx, man_ans], [man_report, man_plot])

        with gr.TabItem("AI Hallucination Detector"):
            gr.Markdown("This tab demonstrates real verification logic on AI-style answers. Supported answers score high; contradicted answers score low.")
            hall_case = gr.Radio(
                choices=[(case["title"], str(index)) for index, case in enumerate(AI_CASES)],
                value="0",
                label="Choose a scenario",
            )
            hall_btn = gr.Button("Run Detection", variant="primary")
            with gr.Row():
                hall_report = gr.Markdown("Detection results will appear here.")
                hall_plot = gr.Plot()
            hall_btn.click(hallucination_demo, [hall_case], [hall_report, hall_plot])

        with gr.TabItem("Quora-Style Answer Ranking"):
            gr.Markdown("Use one question with multiple answers. The app ranks which answer is most supported by the reference context.")
            with gr.Row():
                with gr.Column():
                    rank_case = gr.Radio(
                        choices=[(case["title"], str(index)) for index, case in enumerate(RANKING_CASES)],
                        value="0",
                        label="Load an example",
                    )
                    load_btn = gr.Button("Load Example")
                    rank_q = gr.Textbox(label="Question", lines=2)
                    rank_ctx = gr.Textbox(label="Reference context", lines=6)
                    rank_answers_box = gr.Textbox(
                        label="Answers to rank (one per line)",
                        lines=8,
                        placeholder="Paris is the capital of France.\nBerlin is the capital of France.",
                    )
                    rank_btn = gr.Button("Rank Answers", variant="primary")
                with gr.Column():
                    rank_report = gr.Markdown("Ranking results will appear here.")
                    rank_plot = gr.Plot()
            load_btn.click(load_ranking_case, [rank_case], [rank_q, rank_ctx, rank_answers_box])
            rank_btn.click(rank_answers, [rank_q, rank_ctx, rank_answers_box], [rank_report, rank_plot])




if __name__ == "__main__":
    demo.launch()
