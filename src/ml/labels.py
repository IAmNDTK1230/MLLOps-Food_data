from src.ml.config import LABELS


def label_rating(rating: int, rules: dict[int, str]) -> str:
    label = rules.get(rating)
    if label not in LABELS:
        raise ValueError(f"No valid sentiment label for rating {rating}")
    return label
