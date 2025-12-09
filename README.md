**Trail Analysis — Streamlit App**

- **Project:** Interactive Streamlit app to visualize and analyze trail `.fit` recordings.
- **Files:** `app.py`, `trail_utils.py`, `requirements.txt`.

**Quick Start (Windows PowerShell)**

- Create and activate a virtual environment:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
- Install dependencies:
```
python -m pip install --upgrade pip
pip install -r .\requirements.txt
```
- Run the app:
```
streamlit run .\app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`). Use the sidebar to upload a `.fit` file.

**What the app expects from the FIT file**
- The app parses `record` messages and expects common fields such as `timestamp`, `heart_rate`, `enhanced_speed`, `enhanced_altitude`, `distance`, and `cadence`. Some analyses will be disabled if required columns are missing; the app will show friendly errors.

**Troubleshooting**
- If VS Code reports "Import X could not be resolved", select the workspace interpreter to the project's venv: `Ctrl+Shift+P` → `Python: Select Interpreter` → choose `./.venv/Scripts/python.exe`.
- If upload fails or you see "No records found", verify the FIT file contains `record` messages (some FIT exports only contain sessions or laps).

**Deploy to Streamlit Cloud**
- Push your repository to GitHub. On https://share.streamlit.io create a new app, connect your repo, choose branch `main` and entrypoint `app.py`. Streamlit Cloud will use `requirements.txt` to install dependencies.

**License & Notes**
- This repo is for personal analysis. Adapt the code to your needs and data quirks.
Objectif du projet : - Développer un code permettant d'analyser différents paramètres d'une course 
- Analyser la performance de manière plus poussée qu'avec un logiciel type Nolio
- Créer une app Streamlit pour rendre la création de graphique accessible 
