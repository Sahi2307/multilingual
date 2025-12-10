# Explainable Multilingual Civic Complaint Resolution System

A comprehensive AI-powered civic complaint management system with multilingual support (English, Hindi, Hinglish) and explainable AI using SHAP.

## 🎯 Features

- **Multilingual Support**: Accept complaints in English, Hindi, and Hinglish
- **AI-Powered Classification**: MuRIL for category classification (≥94% accuracy)
- **Urgency Prediction**: XGBoost for 4-level urgency prediction (≥89% accuracy)
- **Explainable AI**: SHAP-based natural-language explanations for all predictions
- **Role-Based Access Control**: Separate pages for Citizens, Officials, and Admins
  - Citizens: File, track, and view complaints
  - Officials: Manage complaints, update status
  - Admins: System analytics and settings
- **Real-time Tracking**: Queue position and status updates
- **Secure Authentication**: Password hashing, session management, multi-role support
- **Interactive Dashboard**: Citizens, Officials, and Admin views with complaint analytics

## 📋 System Requirements

- Python 3.8+
- 4GB+ RAM
- CUDA-capable GPU (optional, for faster training)

## 🚀 Quick Start

### 1. Clone Repository

```bash
cd d:\Downloadss\Project\multilingual
git clone <your-repo-url> civic-complaint-system
cd civic-complaint-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize Database

```bash
python -c "from utils.database import initialize_database; initialize_database()"
```

This creates:
- SQLite database at `data/civic_complaints.db`
- Default admin account: `admin@civiccomplaints.gov` / `Admin@123`

### 5. Generate Training Data

```bash
python src/data_preparation.py
```

This generates 200 synthetic complaints distributed across:
- Categories: Water Supply (67), Sanitation (66), Transportation (67)
- Urgency: Critical (20), High (60), Medium (80), Low (40)
- Languages: Hindi (80), English (60), Hinglish (60)

### 6. Extract Features

```bash
python src/feature_extraction.py
```

This extracts 776-dimensional features:
- 768-dim MuRIL embeddings
- 8-dim structured features

### 7. Train Models

**Category Model (MuRIL):**
```bash
python src/train_category_model.py
```

**Urgency Model (XGBoost):**
```bash
python src/train_urgency_model.py
```

### 8. Run Application

```bash
streamlit run Home.py
```

Access at: http://localhost:8501

## 👥 Default Accounts

### Admin Account
- Email: `admin@civiccomplaints.gov`
- Password: `Admin@123`
- Role: Admin
- **Note**: Change password on first login

### Test Accounts (Create via registration)
- **Citizens**: Self-register to file and track complaints
- **Officials**: Request access; requires admin approval to manage complaints
- **Admins**: Created by existing admins; access to system analytics and admin panel

## 🔐 Access Control

### Citizen Pages
- ✅ Home (landing page)
- ✅ File Complaint (pages/2_File_Complaint.py)
- ✅ My Complaints (pages/3_My_Complaints.py)
- ✅ Track Complaint (pages/4_Track_Complaint.py)
- ✅ About (pages/7_About.py)
- ❌ Cannot access Official Dashboard or Admin Panel

### Official Pages
- ✅ Official Dashboard (pages/5_Official_Dashboard.py)
- ✅ View assigned complaints, update status, add remarks
- ❌ Cannot access citizen complaint pages

### Admin Pages
- ✅ Admin Panel (pages/6_Admin_Panel.py)
- ✅ System analytics, complaint trends, department statistics
- ✅ Approve/manage official accounts
- ❌ Cannot access citizen complaint pages

## 📂 Project Structure

````markdown
# Explainable Multilingual Civic Complaint Resolution System

This repository contains the implementation of a multilingual civic complaint management system using MuRIL (google/muril-base-cased) for category classification and XGBoost for urgency prediction.

## Phase 1: Data Preparation & Model Training

### Setup

1. Create and activate a Python 3.8+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

> On Windows PowerShell you can use:
>
> ```powershell
> python -m pip install -r requirements.txt
> ```

### Directory Structure (current)

- `data/`
  - `raw/` – raw synthetic complaints
  - `processed/` – processed datasets and feature files
- `models/`
  - `muril_category_classifier/` – fine-tuned MuRIL model (created after training)
  - `xgboost_urgency_predictor.pkl` – trained XGBoost urgency model
  - `feature_scaler.pkl` – StandardScaler for urgency features
- `src/`
  - `data_preparation.py` – builds synthetic dataset and structured features
  - `feature_extraction.py` – generates MuRIL embeddings and 776-dim features
  - `train_category_model.py` – fine-tunes MuRIL for 3-class categorization
  - `train_urgency_model.py` – trains XGBoost urgency predictor
- `tests/`
  - `test_data_preparation.py` – validates dataset distributions and language detection
  - `test_feature_extraction.py` – validates 776-dim feature matrices

Additional folders (`pages/`, `utils/`, etc.) will be populated in later phases.

### Running Phase 1

From the project root (`civic-complaint-system/`):

1. **Build the dataset** (200 complaints with splits and structured features):

```bash
python -m src.data_preparation
```

2. **Generate features for urgency model** (MuRIL embeddings + 8 structured features):

```bash
python -m src.feature_extraction
```

3. **Train category model (MuRIL fine-tuning)**:

```bash
python -m src.train_category_model
```

4. **Train urgency model (XGBoost)**:

```bash
python -m src.train_urgency_model
```

Each script logs progress and saves outputs under `data/` and `models/`.

### Testing Phase 1

After you have run the Phase 1 scripts above, you can run the unit tests
for data preparation and feature extraction:

```bash
pytest tests/test_data_preparation.py tests/test_feature_extraction.py
```

These tests verify that the dataset and 776-dimensional feature matrices
match the project specification.

## Phase 2: Explainability Integration

Phase 2 adds SHAP-based explanations on top of the Phase 1 models:

- `src/explainability.py` implements `ExplainabilityEngine`,
  `CategorySHAPExplainer`, and `UrgencySHAPExplainer`.
- `src/test_explainability.py` is a small manual script you can run with:

  ```bash
  python -m src.test_explainability
  ```

  This prints category/urgency predictions and natural-language SHAP
  explanations for the example Hinglish complaint from the project brief.

- `tests/test_explainability_integration.py` contains pytest-based
  integration tests that:
  - Require all Phase 1 scripts to have been run (models + scaler present).
  - Instantiate `ExplainabilityEngine`.
  - Call `explain_category` and `explain_urgency` on the sample complaint.
  - Assert that SHAP outputs are well-formed and consistent with the
    768+8 feature design (776-dimensional SHAP vector, factor importance
    over `text_embedding` and the 8 structured features).

### Running tests for Phases 1 and 2 together

Once you have trained the models (Phase 1), you can run all tests with:

```bash
pytest tests/test_data_preparation.py \
       tests/test_feature_extraction.py \
       tests/test_explainability_integration.py
```

The explainability tests will be automatically skipped if the Phase 1
models have not been trained yet, ensuring a clear separation between
model training and explainability validation.

### Example API response (Phase 1 + Phase 2)

The thin API layer in `src/api.py` exposes a function
`api_register_complaint(payload)` that wires together:

- Phase 1 models (category + urgency prediction).
- Phase 2 explainability (top SHAP keywords and factor importance).

A typical JSON-style response looks like:

```json
{
  "complaint_id": "C1733380000000",
  "category": "Transportation",
  "category_confidence": 0.963,
  "urgency": "Medium",
  "urgency_confidence": 0.893,
  "department": "Municipal Department - Transportation",
  "queue_position": 12,
  "eta_text": "2–3 days",
  "category_top_keywords": [
    "road",
    "kharab",
    "potholes",
    "accident",
    "risk"
  ],
  "urgency_factor_importance": {
    "text_embedding": 41.2,
    "emergency_keyword_score": 15.0,
    "severity_score": 20.1,
    "affected_population": 12.3,
    "text_length": 4.5,
    "repeat_complaint_count": 3.8,
    "hour_of_day": 1.8,
    "is_weekend": 0.7,
    "is_monsoon_season": 0.6
  }
}
```

This illustrates how Phase 1 predictions and Phase 2 SHAP explanations
are combined into a single response that can be consumed by the
Streamlit UI or any future REST API.

## Phase 3: Streamlit Web Application

Phase 3 provides a multi-page Streamlit UI on top of the Phase 1 models
and Phase 2 explainability:

- `Home.py` sets up global configuration, the landing dashboard, and a language selector.
- `pages/2_File_Complaint.py` is the main entry point for citizens:
  - Collects personal and complaint details.
  - Calls `ComplaintProcessor.process_complaint(...)` which uses the
    MuRIL category model, XGBoost urgency model, and SHAP explainers.
  - Displays category and urgency predictions, queue position, ETA, and
    SHAP-based explanations (highlighted text, factor bar chart, waterfall
    and force plots).
- `pages/3_My_Complaints.py` lets citizens enter the email they used when
  filing complaints and:
  - Loads their live complaints from the SQLite database populated by the
    processor (Phase 1+2 outputs).
  - Falls back to the synthetic CSV dataset if no live complaints are
    found.
- `pages/4_Track_Complaint.py` allows users to track a complaint by ID,
  first using the live DB and then the synthetic CSV as a fallback.
- `pages/5_Official_Dashboard.py` shows an official view with:
  - Overview cards for urgency levels using both DB and CSV data.
  - A live complaints table backed by the DB.
  - A side panel where selecting a complaint shows full details, AI
    explanations (via `ComplaintProcessor` and SHAP), and a simple status
    update form that writes back to the `status_updates` table.
- `pages/6_Admin_Panel.py` exposes system-level analytics using both the
  synthetic CSV dataset and live DB.
- `pages/7_About.py` documents how to use the system and explains the AI
  components at a high level.

To run the app (from the project root):

```bash
streamlit run Home.py
```

## Phase 4: Backend Integration & Testing

Phase 4 focuses on wiring the models and explainability into a coherent
backend and validating it end-to-end:

- `src/complaint_processor.py` defines `ComplaintProcessor`, which:
  - Loads the MuRIL category model and XGBoost urgency model.
  - Rebuilds the 8 structured features for new complaints.
  - Predicts category and urgency.
  - Generates SHAP explanations via the `ExplainabilityEngine`.
  - Persists complaints, status updates, notifications, and model
    predictions into the SQLite database.
  - Computes queue position and ETA using a priority score.
- `utils/database.py` defines the SQLite schema and helper functions for
  users, departments, complaints, status updates, notifications and
  model_predictions.
- `src/api.py` provides a thin functional API layer:
  - `api_register_complaint(payload)` registers a complaint via the
    processor and returns a JSON-style summary including category,
    urgency, queue position, ETA, and SHAP summaries.
  - `api_get_complaint_status(complaint_id)` returns complaint details
    and status updates.
  - `api_list_notifications(user_id, include_read=False)` returns stored
    notifications for a user.

### Testing Phase 4

- `tests/test_api_integration.py` exercises the API layer on top of the
  full stack by:
  - Registering a complaint through `api_register_complaint`.
  - Verifying that the complaint is stored in the database and can be
    fetched via `api_get_complaint_status`.
  - Confirming that a notification is recorded and can be retrieved via
    `api_list_notifications`.

Run the Phase 4 tests (after Phase 1 models are trained) with:

```bash
pytest tests/test_api_integration.py
```

### Performance sanity check

For a quick, manual performance check of end-to-end complaint
processing (including SHAP explanations), you can run:

```bash
python -m src.benchmark_processing
```

This script uses `ComplaintProcessor` on a few representative complaints
(English, Hindi, and Hinglish) and prints per-complaint timings and an
average. You can compare the results to the project target of processing
one complaint in under ~4 seconds on your local hardware.

These later phases continue to build on the models and artifacts
produced in Phase 1 and the explainability layer from Phase 2.
````
