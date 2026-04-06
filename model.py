from transformers import pipeline
from tqdm import tqdm


def _normalize_label(name: str) -> str:
    """
    Normalize model output labels into standard categories.

    Args:
        name (str): Raw label from model output.

    Returns:
        str: One of {"positive", "neutral", "negative"} or original label if unknown.
    """
    n = name.lower().strip()
    if "positive" in n:
        return "positive"
    if "negative" in n:
        return "negative"
    if "neutral" in n:
        return "neutral"
    return n


def _probs_to_score_and_label(scores: list[dict]):
    """
    Convert raw model output into structured sentiment metrics.

    Extracts:
    - Top sentiment label
    - Confidence score
    - Probabilities for each class

    Args:
        scores (list[dict]): List of label-score dictionaries from model.

    Returns:
        tuple:
            - sentiment_score (float): Highest probability score
            - pred_label (str): Predicted sentiment label
            - prob_positive (float)
            - prob_neutral (float)
            - prob_negative (float)

    Notes:
        Handles unexpected model formats with fallback logic.
    """
    p_pos = p_neu = p_neg = 0.0
    for d in scores:
        lab = _normalize_label(str(d["label"]))
        prob = float(d["score"])
        if lab == "positive":
            p_pos = prob
        elif lab == "negative":
            p_neg = prob
        elif lab == "neutral":
            p_neu = prob
    # if something other than the current model output format is used
    if p_pos == p_neg == p_neu == 0.0 and scores:
        best = max(scores, key=lambda x: x["score"])
        return 0.0, _normalize_label(str(best["label"])), 0.0, 0.0, float(best["score"])

    score = max(p_pos, p_neg, p_neu)  # get the max score
    if p_pos >= p_neu and p_pos >= p_neg:
        top = "positive"
    elif p_neg >= p_neu:
        top = "negative"
    else:
        top = "neutral"

    return score, top, p_pos, p_neu, p_neg


class SentimentScorer:
    def __init__(self, model_id: str, batch_size: int, device: int):
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device
        self._pipe = self._load_pipeline()

    def _load_pipeline(self):
        return pipeline(
            task="sentiment-analysis",
            model=self.model_id,
            tokenizer=self.model_id,
            top_k=None,  # to get all labels
            truncation=True,
            batch_size=self.batch_size,
            device=self.device,
        )

    def predict(self, texts: list[str]):
        """
        Run batched sentiment inference on a list of texts.
        Args:
            texts (list[str]): List of cleaned text inputs.
        Returns:
            list[tuple]: Each tuple contains:
            (sentiment_score, label, prob_positive, prob_neutral, prob_negative)

        """
        output = []
        with tqdm(total=len(texts), desc="Sentiment analysis", unit="snippet") as pbar:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start: start + self.batch_size]
                raw = self._pipe(batch)
                for item in raw:
                    output.append(_probs_to_score_and_label(item))

                pbar.update(self.batch_size)
        return output
