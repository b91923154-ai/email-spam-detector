# 🛠️ Email & SMS Spam Detector - Complete Tech Stack & Architectural Explanation

> **Purpose of this document:** Easy-to-understand, detailed breakdown of every Machine Learning algorithm, NLP library, Python module, REST API backend, web framework, design pattern, and deployment tool used to build the **Email & SMS Spam Detection System**.

---

## 📚 Table of Contents
1. [High-Level System Architecture](#1-high-level-system-architecture)
2. [Zero-DLL Pure NumPy Machine Learning Engine](#2-zero-dll-pure-numpy-machine-learning-engine)
3. [NLP Preprocessing & Feature Engineering](#3-nlp-preprocessing--feature-engineering)
4. [FastAPI REST Backend & Pydantic Data Transfer Objects](#4-fastapi-rest-backend--pydantic-data-transfer-objects)
5. [Streamlit Web Application & Interactive UI](#5-streamlit-web-application--interactive-ui)
6. [Data Processing & Persistence Tools](#6-data-processing--persistence-tools)
7. [Testing, MLOps & Containerization](#7-testing-mlops--containerization)
8. [Summary Table of Technology Stack](#8-summary-table-of-technology-stack)

---

## 1. High-Level System Architecture

The Email & SMS Spam Detector is built as a **Modular Enterprise ML System**:

```
+-------------------------------------------------------------------+
|                        CLIENT INTERFACES                          |
|    Streamlit Web UI (src/app.py)   |   FastAPI REST API (src/api.py) |
+------------------------------------+------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------+
|                         INFERENCE LAYER                           |
|                      (src/predict.py)                             |
|          Singleton In-Memory Model & Vectorizer Loader            |
+------------------------------------+------------------------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+----------------------------------+           +----------------------------------+
|   TEXT PREPROCESSING MODULE      |           |     ZERO-DLL ML ENGINE MODULE    |
|     (src/preprocessing.py)       |           |          (src/model.py)          |
|  1. Lowercase                    |           | 1. NumPyTfidfVectorizer          |
|  2. NLTK Tokenization            |           |    - TF Sublinear Scaling        |
|  3. Alphanumeric Filter          |           |    - Smooth IDF Weights          |
|  4. Stop-words Removal           |           |    - L2 Vector Normalization     |
|  5. Porter Stemmer               |           | 2. NumPyMultinomialNB            |
|                                  |           |    - Laplace Smoothing (α=0.1)   |
|                                  |           |    - Log-Sum-Exp Softmax         |
+----------------------------------+           +----------------------------------+
                                                                 ^
                                                                 |
                                               +-----------------+----------------+
                                               |     TRAINING PIPELINE MODULE    |
                                               |          (src/train.py)          |
                                               |   Reads dataset (data/spam.csv), |
                                               |   Evaluates & saves model.pkl    |
                                               +----------------------------------+
```

---

## 2. Zero-DLL Pure NumPy Machine Learning Engine

Located in [`src/model.py`](file:///c:/Users/mandeep/OneDrive/Desktop/python/projects/Email-Spam-Detector-master/src/model.py), this custom machine learning engine is built entirely on **NumPy** arrays without external C/C++ dynamic library dependencies (`.dll` files on Windows or `.so` files on Linux).

### Why Zero-DLL Matters
Standard Machine Learning models (`scikit-learn` / `scipy`) rely on compiled C/C++ DLL extensions. Under enterprise security environments like **Windows Defender Application Control (WDAC)** or AppLocker, untrusted DLL binaries can be blocked from running. The pure NumPy engine eliminates this risk completely.

### Component 1: `NumPyTfidfVectorizer`
* **What it does:** Converts raw text strings into numerical matrix vectors using Term Frequency-Inverse Document Frequency (TF-IDF).
* **Mathematical Operations:**
  1. **Vocabulary Building:** Counts document frequencies (`df`) across corpus and selects top $V = 4000$ most frequent terms.
  2. **Sublinear Term Frequency:**
     $$\text{tf}(t, d) = 1 + \ln(\text{count}(t, d)) \quad \text{if count} > 0 \text{ else } 0$$
  3. **Smooth IDF Calculation:**
     $$\text{idf}(t) = \ln \left( \frac{1 + N}{1 + \text{df}(t)} \right) + 1.0$$
  4. **L2 Vector Normalization:** Normalizes row vectors to unit length $\|\mathbf{x}\|_2 = 1.0$ so message length variations do not bias predictions.

### Component 2: `NumPyMultinomialNB`
* **What it does:** Classifies TF-IDF vectors into `spam` (1) or `ham` (0).
* **Mathematical Operations:**
  1. **Laplace Smoothing ($\alpha = 0.1$):** Handles unseen vocabulary terms cleanly:
     $$P(w_i \mid c) = \frac{\sum_{j \in c} X_{j, i} + \alpha}{\sum_k \left( \sum_{j \in c} X_{j, k} + \alpha \right)}$$
  2. **Unnormalized Log Posteriors:**
     $$\text{log\_posterior}(c) = \ln P(c) + \sum_{i} x_i \ln P(w_i \mid c)$$
  3. **Softmax / Log-Sum-Exp Calibration:** Prevents floating-point underflow and converts log posteriors into standard confidence probabilities (0.0% to 100.0%):
     $$P(\text{class} = c \mid \mathbf{x}) = \frac{\exp(\text{log\_posterior}(c) - M)}{\sum_k \exp(\text{log\_posterior}(k) - M)}$$
     where $M = \max_k (\text{log\_posterior}(k))$.

---

## 3. NLP Preprocessing & Feature Engineering

Located in [`src/preprocessing.py`](file:///c:/Users/mandeep/OneDrive/Desktop/python/projects/Email-Spam-Detector-master/src/preprocessing.py):

### Preprocessing Pipeline Sequence:

```
[Raw Text] ---> 1. Lowercase ---> 2. Tokenize ---> 3. Alphanumeric Filter ---> 4. Stopwords Removal ---> 5. Porter Stemmer ---> [Cleaned Tokens]
```

1. **Case Normalization:** Converts all text to lower case (`text.lower()`).
2. **Tokenization:** Uses `nltk.word_tokenize()` with fallback to regex `\b\w+\b`.
3. **Alphanumeric Filtering:** Filters out punctuation, emojis, and symbols (`t.isalnum()`).
4. **Stop-word & Punctuation Removal:** Filters out high-frequency non-informative English words (*'and'*, *'the'*, *'is'*).
5. **Porter Stemming:** Uses NLTK's `PorterStemmer` to strip suffixes (*'winning'*, *'winner'*, *'wins'* $\to$ *'win'*).

---

## 4. FastAPI REST Backend & Pydantic Data Transfer Objects

Located in [`src/api.py`](file:///c:/Users/mandeep/OneDrive/Desktop/python/projects/Email-Spam-Detector-master/src/api.py):

* **FastAPI:** High-performance, async Python REST API framework based on Starlette and OpenAPI standards.
* **Uvicorn:** Production-ready ASGI web server for asynchronous HTTP handling.
* **Pydantic V2 DTOs:** Validates incoming JSON schemas and formats API response structures:
  - `PredictRequest`: Validates single input text length (`min_length=1`).
  - `BatchPredictRequest`: Validates batch input arrays (`min_length=1`, `max_length=100`).
  - `PredictResponse`: Enforces `{label: str, confidence: float}` formatting.
* **CORS Middleware:** Enabled (`CORSMiddleware`) for cross-origin frontend integration.

---

## 5. Streamlit Web Application & Interactive UI

Located in [`src/app.py`](file:///c:/Users/mandeep/OneDrive/Desktop/python/projects/Email-Spam-Detector-master/src/app.py):

* **Streamlit Framework:** Reactive Python web UI library for machine learning applications.
* **Key Features:**
  - **Tab 1: Single Message Analysis:**
    - Quick 1-click sample buttons (*Claim Prize*, *Meeting Invite*, *Bank Fraud*, *Friend Chat*).
    - Color-coded verdict alert boxes (`[SPAM DETECTED]` in red vs `[SAFE (HAM)]` in green).
    - Progress gauge and confidence percentage display.
    - Expandable NLP preprocessing token inspector.
  - **Tab 2: Batch File Processing:**
    - Upload `.csv` or `.txt` message files.
    - Automatic column detection (`text`, `v2`, `message`, `email`, `body`).
    - Metric summary cards (Total Processed, Spam Count, Ham Count, Spam Ratio %).
    - 1-click CSV report exporter (`st.download_button`).

---

## 6. Data Processing & Persistence Tools

* **`pandas` & `numpy`:** Handles dataset loading (`data/spam.csv`), deduplication (dropping 403 duplicate records), 80/20 train-test splitting, and matrix math.
* **`pickle`:** Serializes trained model artifacts (`models/model.pkl` and `models/vectorizer.pkl`) to disk.
* **Singleton Model Loader (`src/predict.py`):** Loads saved pickle models **once** during module import, avoiding disk read overhead on subsequent API calls.

---

## 7. Testing, MLOps & Containerization

### 1. Pytest Test Suite (`tests/`)
Contains **27 automated unit and integration tests** passing in < 3 seconds:
- `test_preprocessing.py`: Tests tokenization, stemming, stop-word removal, and edge inputs.
- `test_custom_model.py`: Tests NumPy vectorizer matrix dimensions, L2 norm, and Laplace smoothing math.
- `test_predict.py`: Integration tests for input validation, confidence ranges, and label outputs.
- `test_api.py`: FastAPI endpoint tests using HTTPX Async Client.

### 2. Multi-Stage Dockerfile (`Dockerfile`)
- Uses lightweight `python:3.11-slim` base image.
- Pre-installs NLTK corpora during Docker build time.
- Exposes port `8000` for FastAPI and `8501` for Streamlit.

---

## 8. Summary Table of Technology Stack

| Category | Component / Library | Purpose in System |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Core development environment |
| **ML Math Engine** | `src/model.py` (NumPy) | Pure NumPy TF-IDF Vectorizer & Multinomial Naive Bayes (Zero-DLL) |
| **NLP Utilities** | `nltk` (`PorterStemmer`, `stopwords`) | Text cleaning, stop-word removal, and word stemming |
| **REST Backend** | `fastapi`, `uvicorn`, `pydantic` | Async REST API endpoints with auto-generated Swagger OpenAPI docs |
| **Web UI** | `streamlit` | Executive web dashboard for single & batch analysis |
| **Data Handling** | `pandas`, `numpy` | Data cleaning, deduplication, and matrix operations |
| **Serialization** | `pickle`, `os` | Model artifact persistence & singleton loading |
| **Testing** | `pytest`, `httpx` | 27 automated unit & integration tests |
| **Deployment** | `Docker` | Multi-stage Docker containerization |
