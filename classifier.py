"""Classify wildfire ignition risk from weather and vegetation indicators."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

FEATURES = ("temperature_c", "humidity_pct", "wind_kph", "dryness_index")


class NearestCentroidClassifier:
    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[str]) -> "NearestCentroidClassifier":
        if not rows or len(rows) != len(labels):
            raise ValueError("Rows and labels must be non-empty and equally sized")
        self.centroids = {}
        for label in sorted(set(labels)):
            members = [row for row, target in zip(rows, labels) if target == label]
            self.centroids[label] = [sum(row[i] for row in members) / len(members) for i in range(len(rows[0]))]
        return self

    def predict_one(self, row: Sequence[float]) -> str:
        return min(self.centroids, key=lambda label: sum((a - b) ** 2 for a, b in zip(row, self.centroids[label])))

    def predict(self, rows: Sequence[Sequence[float]]) -> list[str]:
        return [self.predict_one(row) for row in rows]


def load_csv(path: str | Path) -> tuple[list[list[float]], list[str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [[float(row[name]) for name in FEATURES] for row in rows], [row["risk"] for row in rows]


def accuracy(actual: Sequence[str], predicted: Sequence[str]) -> float:
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def train(path: str | Path) -> tuple[NearestCentroidClassifier, float]:
    rows, labels = load_csv(path)
    model = NearestCentroidClassifier().fit(rows, labels)
    return model, accuracy(labels, model.predict(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict wildfire ignition risk.")
    parser.add_argument("--data", default="data/fire_weather.csv")
    parser.add_argument("--temperature-c", type=float, required=True)
    parser.add_argument("--humidity-pct", type=float, required=True)
    parser.add_argument("--wind-kph", type=float, required=True)
    parser.add_argument("--dryness-index", type=float, required=True)
    args = parser.parse_args()
    model, score = train(args.data)
    risk = model.predict_one([args.temperature_c, args.humidity_pct, args.wind_kph, args.dryness_index])
    print(f"Training accuracy: {score:.1%}")
    print(f"Predicted wildfire risk: {risk}")
