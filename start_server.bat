@echo off
cd /d "%~dp0"
call venv\Scripts\activate
echo Server starting at http://localhost:5000
python server.py
pause
