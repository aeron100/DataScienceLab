# Support & Troubleshooting

**App:** DataScienceLab  
**Version:** 1.0.0  
**Platform:** macOS 14.0+

For bugs or feature requests, please [open an issue on GitHub](https://github.com/aeron100/DataScienceLab-Mac/issues).

---

## Table of Contents

- [File Loading Issues](#file-loading-issues)
- [Preprocessing Issues](#preprocessing-issues)
- [Machine Learning Issues](#machine-learning-issues)
- [AI Chat Issues](#ai-chat-issues)
- [Export Issues](#export-issues)
- [General / Performance](#general--performance)

---

## File Loading Issues

### The app won't open my file
- Supported formats: **CSV, TSV, JSON, Excel (.xlsx), SQLite (.db / .sqlite)**
- Make sure the file extension matches the actual format
- For Excel files, only `.xlsx` is supported (not `.xls`)
- Try saving as CSV from Excel or Numbers and reimporting

### My CSV loads with garbled text / wrong columns
- Ensure the file is **UTF-8 encoded**
- Check that the first row contains column headers
- For TSV files, make sure the delimiter is a tab character (not spaces)

### Excel file shows no data
- Only the **first sheet** of an Excel workbook is loaded
- Verify the sheet contains data starting from cell A1
- Remove any merged cells or header rows above the data

### SQLite file won't load
- The app reads the **first table** found in the database
- Ensure the `.db` or `.sqlite` file is not locked by another process
- Try opening the file with DB Browser for SQLite to verify it's valid

---

## Preprocessing Issues

### KNN imputation is very slow
- KNN imputation is O(n²) and can be slow on datasets with more than ~5,000 rows
- Consider using **Mean**, **Median**, or **Mode** imputation for large datasets

### One-hot encoding created too many columns
- One-hot encoding creates one column per unique value in the selected column
- For high-cardinality columns (many unique values), use **Label Encoding** instead

### Train/test split button is greyed out
- You must have at least 10 rows in your dataset to split
- Select a **Target column** before splitting if using stratified split

---

## Machine Learning Issues

### "You must split your data first" error
- Go to the **Prep** tab and perform a train/test split before training models
- The split must be done after any preprocessing steps you want applied

### Model training produces no results
- Ensure the **Target** column is set and contains valid values
- For classification, the target must have at least 2 unique classes
- For regression, the target must be numeric

### Accuracy or R² shows as 0 or NaN
- Check that your test set is large enough (at least 5–10 rows)
- Verify the target column has no missing values after preprocessing
- For regression tasks, check that the target column contains numeric (not text) values

### Feature importance shows "Not available"
- Feature importance is not available for CreateML models on macOS 14
- Use the **KNN** or **Naive Bayes** algorithms if you need feature importance

### Confusion matrix is blank
- This only appears for **classification** tasks
- Confirm the task type was correctly inferred (check that your target column is categorical)

---

## AI Chat Issues

### Ollama is not connecting
1. Make sure [Ollama](https://ollama.com) is installed and running: open Terminal and run `ollama serve`
2. Verify you have at least one model pulled: `ollama pull llama3`
3. The app connects to `http://localhost:11434` by default — ensure no firewall is blocking it
4. Restart Ollama if it was recently updated

### No models appear in the Ollama model list
- Pull a model first: open Terminal and run `ollama pull llama3` (or any model from [ollama.com/library](https://ollama.com/library))
- Click **Refresh** in the AI Chat tab after pulling

### OpenRouter returns an error
- Double-check your API key is entered correctly (it should start with `sk-or-`)
- Ensure your OpenRouter account has credits or an active subscription
- Some models on OpenRouter require specific account tiers — try a different model

### The AI doesn't know about my data
- Load a dataset and run at least one analysis step before chatting
- The AI context is injected automatically — if you just loaded a file, it should appear in the next message
- Start a **New Chat** after loading a new dataset to reset the context

---

## Export Issues

### Python script export fails
- Ensure you have write permission to the destination folder
- Try exporting to your Desktop or Documents folder

### HTML report is blank or missing sections
- The report only includes sections where analysis has been performed
- Train a model or run EDA first, then export
- Open the HTML file in Safari or Chrome — it is self-contained with inline CSS

### CSV export contains unexpected columns
- The exported CSV reflects the **current processed dataset** (after all preprocessing steps)
- If one-hot encoding was applied, the original column will be replaced by encoded columns

---

## General / Performance

### The app is slow with large datasets
- DataScienceLab is optimized for datasets up to **200,000 rows**
- For datasets above 200,000 rows, consider pre-sampling before loading
- **KNN imputation** uses a 15,000-row reference pool for large datasets — results are approximate but fast
- **DBSCAN** automatically subsamples to 50,000 rows on large datasets and will show a note when this occurs
- Correlation heatmap is the most visually intensive operation but uses vectorized math and remains fast at 200k rows

### The app crashes on launch
- Ensure you are running **macOS 14.0 (Sonoma) or later**
- Try deleting the app and reinstalling from the Mac App Store

### I found a bug not listed here
- Please [open an issue on GitHub](https://github.com/aeron100/DataScienceLab-Mac/issues) with:
  - macOS version
  - Steps to reproduce
  - What you expected vs. what happened
  - A screenshot if applicable
