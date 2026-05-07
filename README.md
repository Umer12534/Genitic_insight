# Genitic_insight

`Genitic_insight` is a Django-based bioinformatics web app inside the `MLDeployment` project. It supports sequence upload, feature extraction, module selection, model training, and report generation.

## Project Structure

```text
ML-Django/
|-- README.md
|-- requirements.txt
|-- myenv/
`-- MLDeployment/
    |-- manage.py
    |-- db.sqlite3
    |-- uploads/
    |-- media/
    |-- MLDeployment/
    `-- Genitic_insight/
```

## Prerequisites

- Windows PowerShell
- Python 3.13 recommended
- `pip`

## Run The Project

Open PowerShell in the repository root:

```powershell
cd "c:\Users\786\OneDrive\group\project relatede\coding files\At The End\ML-Django"
```

### Option 1: Create a fresh virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install reportlab
cd .\MLDeployment
python manage.py migrate
python manage.py runserver
```

### Option 2: Use the existing `myenv` environment

Use this only if that environment already works on your machine.

```powershell
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install reportlab
cd .\MLDeployment
python manage.py migrate
python manage.py runserver
```

After the server starts, open:

```text
http://127.0.0.1:8000/
```

## Important URLs

- Home: `http://127.0.0.1:8000/`
- Feature extraction: `http://127.0.0.1:8000/feature_extraction/`
- Module selection: `http://127.0.0.1:8000/module-selection/`
- About: `http://127.0.0.1:8000/about/`
- Django admin: `http://127.0.0.1:8000/admin/`

## Admin User

If you want to log in to Django admin, create a superuser:

```powershell
cd .\MLDeployment
python manage.py createsuperuser
```

## How This Project Is Configured

- Django project: `MLDeployment/MLDeployment`
- Django app: `MLDeployment/Genitic_insight`
- Main entry file: `MLDeployment/manage.py`
- Database: SQLite (`MLDeployment/db.sqlite3`)
- Uploaded/generated files are stored in `MLDeployment/uploads/`

## Example Input Files

Sample files already in the repo:

- `MLDeployment/media/input.fasta`
- `MLDeployment/media/iedb_linear_epitopes_all.fasta`
- `MLDeployment/media/features_acc.csv`

## Common Commands

Run migrations:

```powershell
cd .\MLDeployment
python manage.py migrate
```

Start development server:

```powershell
cd .\MLDeployment
python manage.py runserver
```

Run Django system checks:

```powershell
cd .\MLDeployment
python manage.py check
```

Create migration files if models change:

```powershell
cd .\MLDeployment
python manage.py makemigrations
python manage.py migrate
```

## Troubleshooting

### PowerShell blocks activation

If PowerShell prevents virtual environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### `ModuleNotFoundError: No module named 'reportlab'`

The code imports `reportlab` in `Genitic_insight/views.py`, but it is not listed in `requirements.txt`. Install it manually:

```powershell
pip install reportlab
```

### Optional ML libraries

`xgboost` and `lightgbm` are optional in this code. The app tries to import them only if they are installed, so the project can still run without them.

### If `python` does not work

Try:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.13 -m pip install -r requirements.txt
```

## Quick Start Summary

```powershell
cd "c:\Users\786\OneDrive\group\project relatede\coding files\At The End\ML-Django"
python -m venv .venv
.\.venv\Scripts\Activate.ps1   #  .\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install reportlab
cd .\MLDeployment
python manage.py migrate
python manage.py runserver
```
