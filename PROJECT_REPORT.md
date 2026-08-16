# Genetic-Insight Project Report

## 1. Project Introduction

**Genetic-Insight** is a Django-based bioinformatics and machine learning web application. The project is located inside the `MLDeployment` Django project and the main Django app is named `Genitic_insight`.

The purpose of this project is to help users upload biological sequence data, extract numerical features from Protein, DNA, or RNA sequences, train machine learning models, evaluate model performance, visualize results, and generate downloadable reports.

In simple words, this project converts biological sequence files into machine learning-ready feature tables and then allows model training and evaluation through a web interface.

## 2. Main Objective

The main objective of the project is to provide an end-to-end platform for biological sequence analysis without requiring the user to manually write Python code.

The application supports:

- Uploading FASTA files.
- Detecting biological sequence type.
- Extracting features from Protein, DNA, and RNA sequences.
- Saving extracted features as CSV data.
- Training machine learning models using uploaded datasets or extracted feature data.
- Evaluating models using common machine learning metrics.
- Displaying ROC and Precision-Recall visualizations.
- Generating model reports in PDF, HTML, or CSV format.

## 3. Project Type

This is a **web-based bioinformatics machine learning application**.

It combines:

- Bioinformatics sequence processing.
- Feature engineering.
- Machine learning model training.
- Model evaluation.
- Data visualization.
- Report generation.

## 4. Project Structure

The main project folder contains:

```text
ML-Django/
|-- myenv/
`-- MLDeployment/
    |-- manage.py
    |-- db.sqlite3
    |-- README.md
    |-- requirements.txt
    |-- media/
    |-- uploads/
    |-- MLDeployment/
    `-- Genitic_insight/
```

Important folders and files:

| Path | Purpose |
| --- | --- |
| `MLDeployment/manage.py` | Django command-line entry point |
| `MLDeployment/db.sqlite3` | SQLite database file |
| `MLDeployment/requirements.txt` | Python package dependencies |
| `MLDeployment/MLDeployment/settings.py` | Django settings and configuration |
| `MLDeployment/MLDeployment/urls.py` | Main project URL routing |
| `MLDeployment/Genitic_insight/views.py` | Main backend logic for pages, feature extraction, training, and reports |
| `MLDeployment/Genitic_insight/models.py` | Database model for uploaded FASTA files |
| `MLDeployment/Genitic_insight/forms.py` | Upload form definitions |
| `MLDeployment/Genitic_insight/urls.py` | App-level URL routes |
| `MLDeployment/Genitic_insight/templates/` | HTML templates |
| `MLDeployment/Genitic_insight/static/` | CSS, JavaScript, and image files |
| `MLDeployment/Genitic_insight/utils/` | Protein, DNA, and RNA feature extraction classes |
| `MLDeployment/uploads/` | Uploaded/generated files, including extracted CSV |
| `MLDeployment/media/` | Example FASTA and CSV files |

## 5. Technologies Used

### Backend

- Python 3.13
- Django 5.1.6
- SQLite
- BioPython
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- ReportLab
- Joblib

### Frontend

- HTML
- CSS
- JavaScript
- Bootstrap
- Bootstrap Icons
- Font Awesome
- Chart.js

### Data and Machine Learning

- FASTA file parsing using BioPython.
- CSV processing using Pandas.
- Numerical operations using NumPy.
- Model training and evaluation using scikit-learn.
- ROC and Precision-Recall charts using Chart.js and Matplotlib.

## 6. Django Configuration

The Django project name is **MLDeployment**.

The Django app name is **Genitic_insight**.

Installed apps include:

- Django admin
- Django auth
- Django sessions
- Django messages
- Django static files
- `Genitic_insight`

The project uses SQLite as the database:

```text
MLDeployment/db.sqlite3
```

Media and uploaded files are configured to use:

```text
MEDIA_ROOT = uploads/
MEDIA_URL = /uploads/
```

The project is currently configured for development:

- `DEBUG = True`
- `ALLOWED_HOSTS = []`
- A hardcoded Django secret key exists in `settings.py`

These settings should be changed before production deployment.

## 7. Application Routes

The project routes all app pages through `Genitic_insight.urls`.

Important URLs:

| URL | View | Purpose |
| --- | --- | --- |
| `/` | `home` | Home page |
| `/feature_extraction/` | `feature_extraction` | Upload and detect sequence type |
| `/feature_extraction/analyze/` | `analyze_sequence` | Extract selected sequence features |
| `/data-visualization/` | `data_visualization` | Data visualization page |
| `/module-selection/` | `module_selection` | ML model training page |
| `/evaluation-values/` | `evaluation_values` | Evaluation page |
| `/train-model/` | `train_model` | Train selected ML algorithms |
| `/module-selection-with-features/` | `module_selection_with_features` | Train using extracted features |
| `/train-model-with-features/` | `train_model_with_features` | Alternative feature training route |
| `/generate_model_report/` | `generate_model_report` | Generate PDF, HTML, or CSV report |
| `/about/` | `about_view` | About page |
| `/admin/` | Django admin | Admin panel |

## 8. Database Model

The project has one custom model:

```python
class FastaFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
```

This model stores uploaded FASTA file information.

Fields:

- `file`: uploaded FASTA file path.
- `uploaded_at`: upload date and time.

The initial migration exists in:

```text
Genitic_insight/migrations/0001_initial.py
```

Note: the model is not currently registered in `admin.py`, so uploaded FASTA records may not appear in Django admin unless registration is added.

## 9. Main User Workflow

The normal user flow is:

1. User opens the home page.
2. User goes to the sequence analysis page.
3. User uploads a FASTA file.
4. The system reads the FASTA sequence using BioPython.
5. The system detects whether the sequence is Protein, DNA, RNA, or Unknown.
6. User selects a descriptor based on the detected sequence type.
7. User starts feature extraction.
8. The backend uses the matching feature extractor class.
9. Extracted features are returned as CSV data.
10. The extracted CSV is saved to `uploads/extracted_CSV.csv`.
11. User can download the CSV.
12. User can train machine learning models using uploaded CSV data or extracted features.
13. Results are displayed with metrics and charts.
14. User can generate a model report.

## 10. Sequence Type Detection

The project detects sequence type in `views.py` using the `detect_sequence_type` function.

The detection checks:

- DNA characters: `A`, `T`, `C`, `G`
- RNA characters: `A`, `U`, `C`, `G`
- Protein amino acid characters
- Stop marker `*`
- Ambiguous nucleotide characters

Possible outputs:

- `DNA`
- `RNA`
- `Protein`
- `Unknown`

## 11. Feature Extraction

Feature extraction is implemented in three main utility classes:

| File | Class | Sequence Type |
| --- | --- | --- |
| `Genitic_insight/utils/proteinfeature.py` | `ProteinFeatureExtractor` | Protein |
| `Genitic_insight/utils/DNAfeature.py` | `DNAFeatureExtractor` | DNA |
| `Genitic_insight/utils/RNAfeature.py` | `RNAFeatureExtractor` | RNA |

### Protein Feature Methods

The protein extractor supports:

- AAC: Amino Acid Composition
- PAAC: Pseudo Amino Acid Composition
- EAAC: Enhanced Amino Acid Composition
- CKSAAP: Composition of k-spaced Amino Acid Pairs
- DPC: Dipeptide Composition
- DDE: Dipeptide Deviation from Expected Mean
- TPC: Tripeptide Composition

### DNA Feature Methods

The DNA extractor supports:

- Kmer
- RCKmer: Reverse Complement K-mer
- Mismatch
- Subsequence
- NAC: Nucleotide Composition
- ANF: Accumulated Nucleotide Frequency
- ENAC: Enhanced Nucleotide Composition

### RNA Feature Methods

The RNA extractor supports:

- Kmer
- Mismatch
- Subsequence
- NAC: Nucleotide Composition
- ENAC: Enhanced Nucleotide Composition
- ANF: Accumulated Nucleotide Frequency
- NCP: Nucleotide Chemical Property
- PSTNPss: Position-specific trinucleotide propensity based on single strand

## 12. Feature Extraction Output

The extracted feature data is converted to CSV.

The backend removes the `ID` column if it exists and saves the latest extraction result to:

```text
MLDeployment/uploads/extracted_CSV.csv
```

The frontend displays the extracted CSV in a paginated table and allows CSV download with a custom filename.

## 13. Machine Learning Training

Machine learning training is handled by the `train_model` function in `views.py`.

The training page supports three data sources:

- Single uploaded dataset with automatic train/test split.
- Separate training and testing CSV files.
- Previously extracted feature data from `uploads/extracted_CSV.csv`.

The target column can be selected by name or index. If the target is categorical, the system uses `LabelEncoder`.

The train/test split uses:

```python
train_test_split(..., random_state=42)
```

## 14. Supported Machine Learning Algorithms

The project supports classification and regression logic.

Frontend algorithm options include:

- Linear Regression
- Logistic Regression
- Random Forest
- Decision Tree
- Support Vector Machine
- K-Nearest Neighbors
- Neural Network

Backend training logic supports:

### Classification

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- Support Vector Classifier
- K-Nearest Neighbors Classifier
- MLP Neural Network Classifier

### Regression

- Linear Regression
- Random Forest Regressor
- Support Vector Regressor

The code also imports or prepares optional support for:

- XGBoost
- LightGBM

However, XGBoost and LightGBM are only conditionally imported and are not fully exposed in the current frontend algorithm list.

## 15. Model Evaluation

For classification, the project calculates:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- ROC curve for binary classification
- ROC AUC
- Precision-Recall curve for binary classification
- Average Precision / AUPRC

For regression, the project calculates:

- Mean Squared Error
- Mean Absolute Error
- R2 score

## 16. Visualization

The project uses Chart.js on the frontend to show:

- ROC curve
- Precision-Recall curve

The backend also uses Matplotlib to generate figures for reports.

## 17. Report Generation

The project can generate model reports from trained model results.

Supported report formats:

- PDF
- HTML
- CSV

The report generation route is:

```text
/generate_model_report/
```

The report can include:

- Metrics
- Charts
- Algorithm parameters

PDF reports are generated using ReportLab.

## 18. Frontend Pages

Important templates:

| Template | Purpose |
| --- | --- |
| `home.html` | Landing/home page |
| `feature_extraction.html` | FASTA upload, sequence detection, feature extraction |
| `module_selection.html` | Dataset upload, algorithm selection, model training, charts, reports |
| `about.html` | Project overview and target users |
| `partials/navbar.html` | Shared navigation |
| `partials/footer.html` | Shared footer |

The frontend uses Bootstrap for layout and styling. JavaScript handles file uploads, AJAX requests, table updates, algorithm selection, model training requests, charts, and report download.

## 19. Static and Media Files

Static files:

```text
Genitic_insight/static/css/
Genitic_insight/static/js/
Genitic_insight/static/images/
```

Important image:

```text
Genitic_insight/static/images/LOGO.png
```

Example media files:

```text
media/input.fasta
media/iedb_linear_epitopes_all.fasta
media/features_acc.csv
```

Generated/uploaded files:

```text
uploads/extracted_CSV.csv
uploads/temp/
```

## 20. Requirements

Important packages from `requirements.txt` include:

- Django
- biopython
- bio
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- joblib
- requests
- pillow
- jupyter packages
- Flask

Important note: `views.py` imports `reportlab`, but `reportlab` is not listed in `requirements.txt`. It should be added to avoid `ModuleNotFoundError` on a fresh setup.

## 21. How to Run the Project

From the project root:

```powershell
cd "C:\Users\786\OneDrive\group\project relatede\coding files\At The End\ML-Django\MLDeployment"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install reportlab
python manage.py migrate
python manage.py runserver
```

Then open:

```text
http://127.0.0.1:8000/
```

## 22. Strengths of the Project

- Provides an end-to-end biological sequence analysis workflow.
- Supports Protein, DNA, and RNA sequences.
- Includes many feature extraction descriptors.
- Allows model training without programming knowledge.
- Supports multiple machine learning algorithms.
- Provides model evaluation metrics.
- Generates visual charts for binary classification.
- Supports downloadable reports.
- Uses Django, which gives a clear web application structure.
- Has a README with setup and run instructions.

## 23. Current Limitations and Issues

The following issues were found during code inspection:

1. `reportlab` is imported but missing from `requirements.txt`.
2. `forms.py` defines `FastaUploadForm` twice. The second definition overwrites the first one.
3. `admin.py` does not register the `FastaFile` model.
4. `tests.py` is empty, so there are currently no automated tests.
5. `DEBUG = True` and the secret key is hardcoded, which is not safe for production.
6. `ALLOWED_HOSTS` is empty, which is fine for local development but not deployment.
7. Some frontend descriptor options, such as `GAAC` and `GAAC_Grouped`, appear in the UI but are not implemented in the backend protein extractor.
8. Some optional ML imports, such as XGBoost and LightGBM, are not fully connected to the frontend training workflow.
9. Uploaded/generated data is stored in a single `extracted_CSV.csv` file, so one user's extraction can overwrite another user's extraction in a multi-user environment.
10. `LinearRegression(normalize=...)` is used in training logic, but newer scikit-learn versions removed the `normalize` parameter.
11. The project contains `__pycache__` files and a local SQLite database, which are usually not committed to a clean repository.
12. Some text in templates contains encoding artifacts such as `â€¢` and `visualizationâ€”all`, which should be corrected.

## 24. Recommended Improvements

Recommended next steps:

- Add `reportlab` to `requirements.txt`.
- Remove the duplicate `FastaUploadForm` definition or rename the forms clearly.
- Register `FastaFile` in Django admin.
- Add unit tests for sequence detection, feature extraction, and model training.
- Move `SECRET_KEY`, `DEBUG`, and other environment-specific settings to environment variables.
- Add proper user/session-based storage for extracted feature files.
- Align frontend descriptor options with backend-supported descriptors.
- Add XGBoost and LightGBM only if they are fully implemented in training and UI.
- Remove unsupported `normalize` parameter from Linear Regression for current scikit-learn versions.
- Add `.gitignore` rules for `__pycache__`, local database files, virtual environments, and generated uploads.
- Improve frontend validation for file size, descriptor parameters, and target column selection.
- Add documentation for accepted FASTA header label format.

## 25. Final Summary

Genetic-Insight is a Django bioinformatics and machine learning platform for biological sequence analysis. It allows researchers or students to upload FASTA files, detect sequence type, extract Protein/DNA/RNA features, train machine learning models, evaluate performance, visualize results, and generate downloadable reports.

The project already has a strong foundation and useful bioinformatics functionality. The most important improvements before production use are dependency cleanup, security configuration, automated tests, better multi-user file handling, and alignment between frontend options and backend-supported methods.
