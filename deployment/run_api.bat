@echo off
REM Laptop 1: start the PathSnap API on all network interfaces, port 8000.
REM Optional: set PATHSNAP_SPLIT=evaluator_test to serve the evaluator trips.
cd /d %~dp0..
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
