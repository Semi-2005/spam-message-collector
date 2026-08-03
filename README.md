<h1 align="center">Spam Message Collector</h1>

<p align="center">
  <strong>An end-to-end Machine Learning pipeline and web application designed to detect spam messages.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.109+-009688.svg?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-19.2+-61DAFB.svg?logo=react" alt="React">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg?logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg?logo=github-actions" alt="CI/CD">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

## 📖 Table of Contents

- [About The Project](#-about-the-project)
  - [Architecture](#architecture)
  - [Built With](#built-with)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [Running the Full Application](#running-the-full-application)
  - [API Inference Example](#api-inference-example)
- [Testing & CI/CD](#-testing--cicd-pipeline)
  - [Running Local Tests](#running-local-tests)
  - [Code Quality & Linting](#code-quality--linting)
  - [GitHub Actions Pipeline](#github-actions-pipeline)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🚀 About The Project

**Spam Message Collector** is a full-stack, end-to-end machine learning system engineered to classify text messages as either spam or safe (ham). It demonstrates a modern, scalable approach to deploying machine learning models, isolating the data pipeline, inference API, and user interface into distinct, manageable components.

### Architecture

The project lifecycle is built on a robust, modular architecture:

1. **Data Collection & Management:** Raw dataset acquisition via automated scripts, securely stored locally. Data is processed and versioned across separate raw and processed data stores.
2. **NLP Pipeline:** Comprehensive text preprocessing utilizing Python and NLTK. This includes lowercasing, punctuation removal, tokenization, stop-word elimination, and lemmatization, followed by TF-IDF vectorization.
3. **Model Training:** Training robust machine learning classifiers (e.g., Naive Bayes, SVM) on the processed corpus. The best-performing model is serialized for decoupled deployment.
4. **Backend (Inference API):** A high-performance FastAPI application. It loads the serialized model on startup (utilizing the Singleton pattern for efficiency) and exposes asynchronous POST endpoints for real-time inference.
5. **Frontend (Client):** A dynamic React.js user interface, allowing users to intuitively input text and visualize the classification results instantly.

### Built With

*   **Core / Data Science:** Python, NLTK, Scikit-Learn, Pandas, NumPy
*   **Backend:** FastAPI, Uvicorn, Pydantic
*   **Frontend:** React.js, Vite, Axios, Oxlint
*   **CI/CD & DevOps:** GitHub Actions, Pytest, Ruff

---

## 📁 Project Structure

```text
spam-message-collector/
├── .github/
│   └── workflows/
│       ├── ci.yml           # Main CI: Python Matrix (3.10-3.12), Pytest, Ruff, Oxlint, Vite Build
│       └── security.yml     # Security and dependency audit workflow
├── backend/                 # FastAPI REST API
├── data/                    # Datasets (raw & processed)
├── frontend/                # React + Vite Client
├── models/                  # ML training scripts and serialized artifacts
├── preprocessing/           # NLP TextCleaner pipeline
├── tests/                   # Pytest unit & API integration test suite
├── pyproject.toml           # Pytest and Ruff configuration
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🧪 Testing & CI/CD Pipeline

The project features a Senior-grade automated CI/CD pipeline powered by **GitHub Actions** with fail-fast execution, dependency caching, and full coverage.

### Running Local Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run full Pytest suite
pytest -v
```

### Code Quality & Linting

```bash
# Python linting and formatting check
ruff check .
ruff format --check .

# Frontend static analysis and production build check
cd frontend
npm run lint
npm run build
```

### GitHub Actions Pipeline

On every `push` and `pull_request` to `main`, GitHub Actions automatically:
1. 🐍 **Python Quality Gate:** Runs `ruff check` and `ruff format` to enforce clean code.
2. 🧪 **Matrix Testing:** Executes 20+ unit and API integration tests across Python `3.10`, `3.11`, and `3.12`.
3. ⚛️ **Frontend Linter:** Performs static analysis with `oxlint`.
4. 📦 **Production Build:** Validates that the React/Vite SPA builds successfully without errors.
5. ⚡ **Smart Caching:** Utilizes `pip` and `npm` cache to maintain pipeline speeds under 30 seconds.

---

## 🗺 Roadmap

- [ ] Implement advanced Deep Learning models (e.g., LSTM, BERT).
- [x] Add extensive unit and integration tests (20+ automated tests).
- [ ] **Dockerization:** Containerize the React frontend and FastAPI backend using Docker and `docker-compose` for seamless, independent scaling and deployment.
- [x] Implement CI/CD pipelines with GitHub Actions.

---

## 🛠 Getting Started

Follow these steps to set up the project locally from scratch.

### Prerequisites

*   **Python 3.10+** ([python.org](https://www.python.org/downloads/))
*   **Node.js v18+** and **npm** ([nodejs.org](https://nodejs.org/))
*   **Git** ([git-scm.com](https://git-scm.com/))

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Semi-2005/spam-message-collector.git
    cd spam-message-collector
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate        # macOS / Linux
    # .venv\Scripts\activate         # Windows
    ```

3.  **Install backend (Python) dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    > **Not:** NLTK veri dosyaları (stopwords, wordnet, punkt) ilk çalıştırmada otomatik indirilir. İnternet bağlantısı gereklidir.

4.  **Download the dataset:**
    ```bash
    python download_dataset.py
    ```

5.  **Train the ML model:**
    ```bash
    python -m models.train_model
    ```
    > Bu adım `models/artifacts/` altına eğitilmiş model (`best_model_pipeline.joblib`) ve metadata dosyasını oluşturur. Eğitim ~30 saniye sürer.

6.  **Install frontend (Node.js) dependencies:**
    ```bash
    cd frontend
    npm install
    cd ..
    ```

---

## 💻 Usage

### Running the Full Application

The application requires **two terminals** — one for the backend API and one for the frontend React app.

**Terminal 1 — Backend (FastAPI):**

```bash
cd spam-message-collector
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

> The API will be available at `http://127.0.0.1:8000`. Swagger docs: `http://127.0.0.1:8000/docs`

**Terminal 2 — Frontend (React):**

```bash
cd spam-message-collector/frontend
npm run dev
```

> Frontend `http://localhost:5173` adresinde açılacaktır.

**Tarayıcıda** `http://localhost:5173` adresini açarak uygulamayı kullanabilirsiniz.

### API Inference Example

You can also test the classification endpoint directly using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "Congratulations! You have won a $1,000 Walmart gift card. Click here to claim your prize."}'
```

**Expected Response:**
```json
{
  "text": "Congratulations! You have won a $1,000 Walmart gift card. Click here to claim your prize.",
  "label": "spam",
  "is_spam": true,
  "spam_probability": 0.671,
  "confidence_level": "Orta"
}
```

---

## 🗺 Roadmap

- [ ] Implement advanced Deep Learning models (e.g., LSTM, BERT).
- [ ] Add extensive unit and integration tests.
- [ ] **Dockerization:** Containerize the React frontend and FastAPI backend using Docker and `docker-compose` for seamless, independent scaling and deployment.
- [ ] Implement CI/CD pipelines with GitHub Actions.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.