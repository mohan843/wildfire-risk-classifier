import unittest
from pathlib import Path

from classifier import NearestCentroidClassifier, accuracy, load_csv, train

DATA = Path(__file__).parents[1] / "data" / "fire_weather.csv"


class WildfireTests(unittest.TestCase):
    def test_dataset_loads(self):
        rows, labels = load_csv(DATA)
        self.assertEqual(len(rows), 18)
        self.assertEqual(set(labels), {"low", "medium", "high"})

    def test_classifier_learns_weather_profiles(self):
        model, score = train(DATA)
        self.assertGreaterEqual(score, 0.80)
        self.assertEqual(model.predict_one([18, 72, 8, 18]), "low")
        self.assertEqual(model.predict_one([35, 28, 30, 82]), "high")

    def test_prediction_count(self):
        rows, _ = load_csv(DATA)
        model, _ = train(DATA)
        self.assertEqual(len(model.predict(rows)), 18)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            NearestCentroidClassifier().fit([], [])

    def test_accuracy(self):
        self.assertEqual(accuracy(["low", "high"], ["low", "low"]), 0.5)


if __name__ == "__main__":
    unittest.main()
