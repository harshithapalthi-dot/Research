# AI Research Paper Category Classifier

**IBM Edunet Foundation Internship Project**

---

## Project Description

This project uses machine learning and NLP techniques to automatically classify research papers into their subject categories based on the paper's **title** and **abstract/summary**. It is built as a fully interactive web application using Streamlit and trained on the real arXiv scientific dataset.

---

## Dataset

**File:** `data/arXiv_scientific_dataset.csv`

The actual provided arXiv scientific dataset is used — **no dummy or synthetic data**. The dataset contains ~287,000 research papers with fields including `id`, `title`, `category`, `category_code`, `published_date`, `updated_date`, `authors`, `first_author`, `summary`, and `summary_word_count`.

- **Target column:** `category` (146 unique research categories)
- **Problem type:** Multiclass Classification
- **Text features used:** `title` + `summary` (combined)

---

## Technologies

| Technology     | Purpose                                      |
|---------------|----------------------------------------------|
| Python 3.10+  | Core language                                |
| Pandas        | Data loading, cleaning, inspection           |
| NumPy         | Numerical operations                         |
| Scikit-learn  | TF-IDF, model training, evaluation           |
| TF-IDF        | Text feature extraction (vectorization)      |
| Streamlit     | Web application framework                    |
| Joblib        | Model serialization / persistence            |

---

## Machine Learning Workflow

```
Dataset (arXiv_scientific_dataset.csv)
  → Data Inspection       (shape, types, nulls, class distribution)
  → Data Cleaning         (drop nulls, deduplicate, remove tiny classes)
  → Sampling Strategy     (per-class cap to manage memory — see below)
  → Text Preprocessing    (lowercase, whitespace normalization)
  → TF-IDF Vectorization  (unigrams + bigrams, max 80k features)
  → Train / Test Split    (80 / 20, stratified)
  → Multiple Model Training
      ├─ Logistic Regression
      ├─ Linear SVM (Calibrated)
      └─ Multinomial Naive Bayes
  → Model Comparison      (Accuracy, Precision, Recall, F1-weighted)
  → Best Model Selection  (automatic — highest weighted F1)
  → Save Pipeline         (models/model.pkl + models/metadata.json)
  → Streamlit Prediction  (load once, predict on demand)
```

---

## Evaluation Metrics

All models are evaluated with **weighted averaging** for Precision, Recall, and F1-score. Weighted averaging accounts for class imbalance by weighting each class's score by its support (number of true samples), which is appropriate for this dataset because the category distribution is highly skewed (e.g., Machine Learning has ~80k papers while some categories have fewer than 50).

---

## Sampling / Large-Dataset Strategy

The full dataset has ~287,000 rows across 146 categories. To make training practical on standard hardware (laptops, Streamlit Cloud free tier), `train_model.py` applies a **per-class stratified cap**:

- Each category contributes at most `MAX_SAMPLES_PER_CLASS = 2000` rows.
- Categories with fewer rows keep all their data.
- Categories with fewer than `MIN_CLASS_SAMPLES = 10` rows are removed (cannot be reliably split).
- The resulting training set still uses the **actual arXiv data** — no synthetic samples are generated.
- The `MAX_SAMPLES_PER_CLASS` constant can be raised for higher accuracy at the cost of more memory/time.

---

## Project Structure

```
project/
│
├── app.py                         # Streamlit web application
├── train_model.py                 # Dataset inspection + ML training script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── agent_instructions.md          # Agent / developer instructions
├── .env.example                   # Environment variable template
│
├── data/
│   └── arXiv_scientific_dataset.csv   # Actual arXiv dataset
│
└── models/
    ├── model.pkl                  # Saved trained pipeline (generated)
    └── metadata.json              # Model metadata (generated)
```

---

## How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This will:
- Inspect the dataset and print a summary
- Clean the data
- Apply the sampling strategy
- Train and compare three models
- Save the best model to `models/model.pkl`
- Save metadata to `models/metadata.json`

Expected output includes a comparison table and the selected best model.

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Streamlit Community Cloud Deployment

1. Push this project to a **public GitHub repository** (include `data/arXiv_scientific_dataset.csv` or host it via Git LFS).

   > **Note:** If the CSV exceeds GitHub's 100 MB file limit, use [Git LFS](https://git-lfs.github.com/) or commit `models/model.pkl` directly (run `train_model.py` locally first, then commit the generated `model.pkl` and `metadata.json`).

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. Click **New app**, select your repository, branch `main`, and set the main file to `app.py`.

4. Click **Deploy**. Streamlit Cloud will run `pip install -r requirements.txt` automatically.

   - If committing `model.pkl` directly, training is not required on the cloud.
   - If you prefer cloud training, ensure `data/arXiv_scientific_dataset.csv` is present in the repo.

5. The app will be live at `https://<your-app>.streamlit.app`.

**All paths are relative** — no absolute paths are used anywhere, so the project works identically on any machine and on Streamlit Cloud.

---

## Notes

- No API keys are required for the core application.
- The model is loaded once per session (cached) for performance.
- Confidence scores are displayed when the model supports `predict_proba`.
- The best model is selected automatically at training time — not hard-coded.
