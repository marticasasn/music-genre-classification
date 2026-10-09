# Music Genre Classification

Group project for ELEC3612 (Pattern Recognition and Machine Intelligence) at the University of Sydney.

Classifies 30-second music clips into 12 genres with a CNN on mel-spectrograms, a BiLSTM on MFCC sequences, and an SVM baseline, then compares all three on the same held-out test set.

## Why

Genre is a fuzzy label: the same track can sound like rock, punk or pop depending on who you ask. This project asks a concrete question about that problem: does treating audio as an **image** (spectrogram + CNN) or as a **time series** (MFCC frames + recurrent network) work better, and does either beat a classical baseline?

## Results

Test set: 2,645 clips, 12 classes, stratified split.

| Model | Input | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|---|
| **CNN 2D** | Mel-spectrogram (1 x 128 x 128) | **0.506** | **0.507** | **0.487** |
| SVM (RBF) | Pooled MFCC stats (80-dim) | 0.481 | 0.475 | 0.437 |
| BiLSTM | MFCC sequence (130 x 40) | 0.404 | 0.400 | 0.351 |

Uniform random guessing scores about 0.08. The CNN wins on every aggregate metric. Classical and Hip-Hop are among the easiest genres for all three models, while Jazz and Pop are consistently among the hardest. Full metrics live in [`results/final_results.json`](results/final_results.json).

![Per-class F1 comparison](results/figures/per_class_f1_comparison.png)

<!-- TODO: add a demo GIF or extra screenshots here (e.g. confusion matrices side by side) -->
<!-- ![Demo](docs/demo.gif) -->

## How it works

1. **Data.** [`lewtun/music_genres`](https://huggingface.co/datasets/lewtun/music_genres) on HuggingFace: 24,985 clips, 30 s each, 22,050 Hz. Eight ambiguous or under-represented genres are dropped, leaving 17,632 clips across 12 genres: Chiptune / Glitch, Classical, Country, Electronic, Experimental, Folk, Hip-Hop, Instrumental, Jazz, Pop, Punk, Rock.
2. **Split.** Stratified 70 / 15 / 15 (12,342 train, 2,645 val, 2,645 test). Normalisation statistics come from the train split only.
3. **Features.** Log-scaled mel-spectrograms resized to 128 x 128, and 40-coefficient MFCCs over 130 frames.
4. **Models.**
   - **CNN 2D:** 4 x [Conv2d, BatchNorm, ReLU, MaxPool], adaptive average pooling, then a 512-unit dense head with dropout.
   - **BiLSTM:** 2 layers, 256 hidden units per direction; the final forward and backward states are concatenated and passed to a dense head.
   - **SVM:** RBF kernel (C=10, balanced class weights) on per-coefficient MFCC mean and standard deviation, standardised.
5. **Training.** Adam, class-weighted cross-entropy to handle imbalance, gradient clipping, `ReduceLROnPlateau`, and early stopping on validation accuracy.
6. **Evaluation.** Accuracy, weighted and macro F1, per-class F1, row-normalised confusion matrices and macro-averaged one-vs-rest ROC curves.

## Stack

Python 3.10+, PyTorch, librosa, scikit-learn, HuggingFace `datasets`, NumPy, pandas, matplotlib, seaborn. Training runs on Kaggle GPUs.

## Getting started

### Option A: Kaggle (recommended)

The training notebooks are written for Kaggle and clone this repo at runtime to import `src/`.

1. Upload a notebook from `notebooks/` to Kaggle and enable a GPU.
2. Click **Add Data** and attach the preprocessed tensors: [pr-a2-preprocesseddataset](https://www.kaggle.com/datasets/marticasas/pr-a2-preprocesseddataset) (about 1.45 GB). It mounts at `/kaggle/input/pr-a2-preprocesseddataset/`.
3. Run the notebooks in order:

| Step | Notebook | What it does |
|---|---|---|
| 1 | `01_eda_kaggle.ipynb` | Exploratory analysis and genre filtering |
| 2 | `02_preprocessing_kaggle.ipynb` | Extracts spectrograms and MFCCs, exports tensors (skip if you use the Kaggle dataset) |
| 3 | `03_cnn_kaggle.ipynb` | Trains the CNN, saves `best_cnn.pth` and `cnn_predictions.csv` |
| 4 | `04_lstm_kaggle.ipynb` | Trains the BiLSTM, saves `best_lstm.pth` and `lstm_predictions.csv` |
| 5 | `05_evaluation_kaggle.ipynb` | Trains the SVM and compares all three models |

Notebook 05 reads the outputs of steps 3 and 4. Save each run's output as a Kaggle dataset named `cnn-results` and `lstm-results` and attach both before running it.

### Option B: local

```bash
git clone https://github.com/marticasasn/music-genre-classification.git
cd music-genre-classification
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: download the raw dataset (~10 GB) into music_genres_local/
python download_dataset.py
```

Then open the notebooks with `jupyter notebook`. The `/kaggle/...` paths at the top of each notebook need to point to your local copies.

## Project structure

```
.
├── notebooks/
│   ├── 01_eda_kaggle.ipynb            # EDA and genre filtering
│   ├── 02_preprocessing_kaggle.ipynb  # Feature extraction and tensor export
│   ├── 03_cnn_kaggle.ipynb            # CNN training
│   ├── 04_lstm_kaggle.ipynb           # BiLSTM training
│   └── 05_evaluation_kaggle.ipynb     # SVM baseline and model comparison
├── src/
│   ├── dataset.py     # SpectrogramDataset, MFCCDataset, class weights
│   ├── models.py      # CNNClassifier, LSTMClassifier
│   ├── train.py       # Training loop, early stopping, checkpointing
│   └── evaluate.py    # Metrics, confusion matrices, training curves, ROC
├── results/
│   ├── figures/       # EDA plots, confusion matrices, curves
│   ├── *_metrics.json, final_results.json, comparative_results.csv
│   └── *_predictions.csv
├── docs/
│   └── assignment2_roadmap.html       # Project plan
├── Report.pdf                         # Final technical report
├── main_v5_current.tex                # LaTeX source of the report
├── download_dataset.py                # Fetches the raw dataset from HuggingFace
└── requirements.txt
```
