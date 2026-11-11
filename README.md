# Wildfire Risk Classifier

A geospatial and environmental machine-learning mini-project that classifies wildfire ignition risk as low, medium, or high from temperature, humidity, wind speed, and a vegetation-dryness index.

## Overview

Fire-management teams use weather and fuel conditions to prioritize monitoring. This project demonstrates an interpretable nearest-centroid classifier over compact fire-weather features.

## Problem statement

Given local weather and dryness indicators, predict a broad ignition-risk category for educational screening and prioritization.

## Features

- Nearest-centroid classification from scratch.
- Multivariate Euclidean distance over fire-weather features.
- Reproducible CSV data and command-line inference.
- Unit tests and training accuracy reporting.
- Standard-library-only implementation.

## Technology stack

- Python 3.10+
- Environmental and geospatial ML concepts
- Nearest-centroid classification
- CSV, `argparse`, and `pathlib`
- `unittest` for automated testing

## Installation

```bash
git clone https://github.com/mohan843/wildfire-risk-classifier.git
cd wildfire-risk-classifier
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party runtime dependencies are required.

## Execution

```bash
python classifier.py --temperature-c 18 --humidity-pct 72 --wind-kph 8 --dryness-index 18
```

High-risk example:

```bash
python classifier.py --temperature-c 35 --humidity-pct 28 --wind-kph 30 --dryness-index 82
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## Sample output

```text
Training accuracy: 100.0%
Predicted wildfire risk: low
```

## Responsible use and limitations

The data is synthetic and the classifier is an educational screening example. It must not replace official fire-danger indices, satellite analysis, emergency-management systems, evacuation guidance, or local authorities. Real deployments need spatial context, forecast uncertainty, fuel maps, and careful false-negative analysis.

## Future improvements

- Add satellite vegetation indices, elevation, and land-cover features.
- Compare calibrated probabilistic and ensemble models.
- Add geospatial cross-validation and spatial drift monitoring.
- Integrate weather forecasts and live sensor feeds.
- Build a map-based monitoring dashboard with alert history.
