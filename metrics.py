import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from openai import OpenAI



### Metric code adapted from https://wandb.ai/ai-team-articles/llm-evaluation/reports/LLM-evaluation-benchmarking-Beyond-BLEU-and-ROUGE--VmlldzoxNTIzMTY0NQ ###


def bleu_score(reference: str, output: str) -> dict:
    """
    Statistical metric: measures n-gram overlap between output and reference.
    Good for: detecting catastrophic failures, regression testing.
    Bad for: paraphrased but correct answers.
    """
    score = sacrebleu.sentence_bleu(
        output,
        [reference],
        smooth_method="exp"
    ).score
    return {"bleu": score / 100.0}


def rouge_l_score(reference: str, output: str) -> dict:
    """
    Statistical metric: measures longest common subsequence.
    Good for: summarization tasks, keyword preservation.
    Bad for: restructured but semantically identical answers.
    """
    _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = _rouge.score(reference, output)
    return {"rouge_l": scores["rougeL"].fmeasure}


def rouge_1_scores(reference: str, output: str) -> dict:

    _rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = _rouge.score(reference, output)
    return {"rouge_1": scores["rouge1"].fmeasure}


def rouge_2_scores(reference: str, output: str) -> dict:

    _rouge = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
    scores = _rouge.score(reference, output)
    return {"rouge_2": scores["rouge2"].fmeasure}


def bert_score(reference: str, output: str) -> dict:
    """
    Semantic metric: uses BERT embeddings to measure similarity.
    Good for: recognizing paraphrases, semantic equivalence.
    Bad for: factual correctness (can score high on fluent nonsense).
    """
    try:
        _, _, F1 = bert_score_fn(
            [output],
            [reference],
            lang="en",
            verbose=False
        )
        return {"bert_score": float(F1[0])}
    except Exception as e:
        print(f"BERTScore failed: {e}")
        return {"bert_score": None}


ACCURACY_JUDGE_PROMPT = """You are evaluating the factual accuracy of an answer.


Question: {question}
Reference Answer: {reference}
Model Answer: {output}


Rate the factual accuracy on a scale of 1-5:
1 = Completely incorrect or contradicts the reference
2 = Mostly incorrect with minor correct elements
3 = Partially correct but missing key information
4 = Mostly correct with minor issues
5 = Completely accurate and equivalent to the reference


Consider:
- Are the core facts correct?
- Does it contradict the reference answer?
- Is critical information missing?


Respond with ONLY a single number (1-5)."""


def accuracy_judge(question: str, reference: str, output: str) -> dict:
    """
    LLM-as-a-judge metric focusing on factual correctness.
    Uses GPT-4o-mini for cost-effective evaluation.
    """
    openai_client = OpenAI()

    prompt = ACCURACY_JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        output=output
    )
    res = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    try:
        score = float(res.choices[0].message.content.strip())
        # Normalize to 0-1 range, cap at 0.9999 for consistent percentage display
        normalized_score = min((score - 1) / 4, 0.9999)
    except ValueError:
        normalized_score = None
    return {
        "accuracy_score": normalized_score,
        "accuracy_raw": score if normalized_score is not None else None
    }



HELPFULNESS_JUDGE_PROMPT = """You are evaluating how helpful an answer is to a user.


Question: {question}
Model Answer: {output}


Rate the helpfulness on a scale of 1-5:
1 = Not helpful at all, confusing or wrong
2 = Minimally helpful, lacks important context
3 = Somewhat helpful, adequate but could be clearer
4 = Helpful, clear and addresses the question well
5 = Extremely helpful, clear, complete, and well-explained


Consider:
- Does it directly address the question?
- Is it clear and easy to understand?
- Does it provide enough context/explanation?
- Would this satisfy a user asking this question?


Respond with ONLY a single number (1-5)."""


def helpfulness_judge(question: str, output: str) -> dict:
    """
    LLM-as-a-judge metric focusing on user helpfulness.
    Uses GPT-3.5-turbo for different perspective and lower cost.
    """
    openai_client = OpenAI()

    prompt = HELPFULNESS_JUDGE_PROMPT.format(
        question=question,
        output=output
    )
    res = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    try:
        score = float(res.choices[0].message.content.strip())
        normalized_score = (score - 1) / 4
    except ValueError:
        normalized_score = None
    return {
        "helpfulness_score": normalized_score,
        "helpfulness_raw": score if normalized_score is not None else None
    }








