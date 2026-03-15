from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from nlp_utils import repair_mojibake_deep


DEFAULT_INTENTS_PATH = Path(__file__).resolve().parent / "data" / "chat data" / "conversation_intents.json"
DEFAULT_MODEL_OUTPUT = Path(__file__).resolve().parent / "model_conversation_intent.pkl"


def _dataset_text(value: Any) -> str:
    if isinstance(value, dict):
        en = str(value.get("en", "")).strip()
        ar = str(value.get("ar", "")).strip()
        return " ".join([v for v in (en, ar) if v]).strip()
    return str(value or "").strip()


def _load_training_pairs(intents_path: Path, include_responses: bool) -> list[tuple[str, str]]:
    raw = json.loads(intents_path.read_text(encoding="utf-8"))
    payload = repair_mojibake_deep(raw)
    intents = payload.get("intents", []) if isinstance(payload, dict) else []

    pairs: list[tuple[str, str]] = []
    for item in intents:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", "")).strip().lower()
        if not tag:
            continue

        patterns = item.get("patterns", [])
        if isinstance(patterns, list):
            for p in patterns:
                text = _dataset_text(p)
                if text:
                    pairs.append((text, tag))

        if include_responses:
            responses = item.get("responses", [])
            if isinstance(responses, list):
                for r in responses:
                    text = _dataset_text(r)
                    if text:
                        pairs.append((text, tag))

    return pairs


def train_and_save(
    intents_path: Path,
    output_path: Path,
    include_responses: bool,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    pairs = _load_training_pairs(intents_path, include_responses=include_responses)
    if not pairs:
        raise ValueError(f"No usable intents found in: {intents_path}")

    texts, labels = zip(*pairs)
    labels_list = list(labels)

    if len(set(labels_list)) >= 2 and len(labels_list) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            list(texts),
            labels_list,
            test_size=test_size,
            random_state=random_state,
            stratify=labels_list,
        )
    else:
        X_train, y_train = list(texts), labels_list
        X_test, y_test = [], []

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)

    metrics = {}
    if X_test:
        y_pred = pipeline.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        }

    artifact = {
        "model": pipeline,
        "model_name": "tfidf_logistic_regression",
        "labels": sorted(set(labels_list)),
        "metrics": metrics,
        "dataset_path": str(intents_path),
        "dataset_rows": int(len(labels_list)),
        "include_responses": bool(include_responses),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(artifact, f)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train intent classifier from conversation_intents.json.")
    parser.add_argument("--intents", type=Path, default=DEFAULT_INTENTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--include-responses", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    artifact = train_and_save(
        intents_path=args.intents,
        output_path=args.output,
        include_responses=args.include_responses,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("Conversation intent model trained successfully")
    print(f"Rows: {artifact['dataset_rows']}")
    print(f"Labels: {artifact['labels']}")
    if artifact.get("metrics"):
        print(f"Accuracy: {artifact['metrics'].get('accuracy', 0):.4f}")
        print(f"Weighted F1: {artifact['metrics'].get('weighted_f1', 0):.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
