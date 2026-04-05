@echo off
echo Stopping server at http://localhost:5000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000') do taskkill /F /PID %%a
echo Server stopped.
pause
