import evaluate
import bert_score

def _bleu(preds: list[str], refs: list[list[str]]) -> float:
    """
    Compute BLEU score
    """
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=preds, references=refs, smooth=True)

def _bert_score(preds: list[str], refs: list[list[str]]) -> float:
    """
    Compute BERTScore
    """
    return bert_score.score(cands=preds, refs=refs, model_type='microsoft/deberta-large-mnli')

def _mc_accuracy(preds: list[str], refs: list[str]):
    """
    Compute Multiple Choice Accuracy
    """
    return sum([pred == ref for pred, ref in zip(preds, refs)]) / len(preds)

def _da_accuracy(preds: list[str], refs: list[list[str]]):
    """
    Compute Direct Answer Accuracy
    """
    return sum([1 if ref.count(pred) >= 3 else 0 for pred, ref in zip(preds, refs)]) / len(preds)