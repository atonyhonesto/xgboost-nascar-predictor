@echo off
rem Double-click to open the predictor dashboard in your browser.
cd /d "%~dp0"
".venv\Scripts\streamlit.exe" run app.py
