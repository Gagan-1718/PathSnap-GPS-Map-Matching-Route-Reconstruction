@echo off
REM Laptop 2: serve the evaluator UI at http://localhost:5500
cd /d %~dp0..\ui
python -m http.server 5500
