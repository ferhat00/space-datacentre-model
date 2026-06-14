"""Streamlit front-end for the orbital data centre viability model (odc package).

This is an *application*, not part of the importable `odc` library. Run it with::

    pip install -e .[app]
    streamlit run app.py

It binds sliders to the shared `odc.schema.GROUPS` schema and drives the OO model
(`odc.model.ODCModel`), so it stays in lock-step with the Chart.js dashboard and the
calibrated kernel.
"""
