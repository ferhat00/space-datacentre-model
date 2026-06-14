"""Entrypoint for the orbital data centre Streamlit app.

    pip install -e .[app]      # or: pip install -r requirements.txt
    streamlit run app.py

The app is additive: it shares the calibrated `odc` package with the existing Chart.js
dashboard (build_dashboard.py) and the analysis notebook (build_notebook.py).
"""
from streamlit_app.main import run

run()
