# DataScienceLab

A native macOS data science workbench for exploring, analyzing, and modeling tabular data — no code required.

![Platform](https://img.shields.io/badge/platform-macOS%2014%2B-blue)
![Version](https://img.shields.io/badge/version-1.0.0-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

DataScienceLab is a fully native SwiftUI application built for macOS 14+. It gives data analysts, researchers, students, and developers a fast, private, offline-capable environment for end-to-end data science workflows.

---

## Features

### Data Ingestion
- Load **CSV, TSV, JSON, Excel (.xlsx), and SQLite** files
- Drag-and-drop or file picker import
- 4 built-in sample datasets (Titanic, Iris, Housing, E-Commerce)
- Automatic column type detection, null statistics, and quartile summaries

### Preprocessing
- **Missing values:** 7 strategies including mean, median, mode, constant, forward fill, backward fill, and KNN imputation
- **Encoding:** Label encoding and one-hot encoding
- **Scaling:** Standard, Min-Max, and Robust scaling
- **Splitting:** Stratified train/test split with seeded shuffle
- Drop and rename columns

### Exploratory Data Analysis (EDA)
- Histograms, value counts, correlation heatmaps (Pearson)
- Box plots, scatter charts, and outlier detection
- Canvas-based correlation matrix for large column sets

### Supervised Machine Learning
- **Algorithms:** Boosted Trees, Random Forest, Decision Tree, Logistic Regression, Linear Regression, KNN, Gaussian Naive Bayes
- **Classification metrics:** Accuracy, F1, Precision, Recall, Confusion Matrix, AUC/ROC
- **Regression metrics:** MAE, RMSE, R², MAPE
- Feature importance visualization

### Unsupervised Machine Learning
- **K-Means** with k-means++ initialization
- **DBSCAN** clustering
- **PCA** with scree plot and PC loadings table
- **Isolation Forest** anomaly detection

### Export
- Export cleaned data as **CSV**
- Generate a runnable **Python / scikit-learn script**
- Generate a self-contained **HTML report**

### AI Assistant
- Built-in chat powered by **Ollama** (fully local, no data leaves your machine)
- **OpenRouter** support for cloud models (API key stored securely in Keychain)
- Session context automatically injected — the assistant knows your dataset, pipeline, and model results

---

## Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon or Intel Mac
- For AI Chat (Ollama): [Ollama](https://ollama.com) installed and running locally
- For AI Chat (OpenRouter): An [OpenRouter](https://openrouter.ai) API key

---

## Getting Started

1. Download DataScienceLab from the **Mac App Store**
2. Open the app and go to the **Data** tab
3. Click **Load File** or drag a CSV/Excel/SQLite file onto the window
4. Work through the tabs: **Prep → EDA → Models → Unsupervised → Export**
5. Optionally open the **AI Chat** tab and connect Ollama or enter your OpenRouter key

---

## Privacy

DataScienceLab does not collect, transmit, or store any of your data. All processing happens on-device. See [PRIVACY.md](PRIVACY.md) for full details.

---

## Support

Having trouble? See [SUPPORT.md](SUPPORT.md) for troubleshooting steps and how to reach us.

---

## License

MIT License. See `LICENSE` for details.
