@echo off
rem Starts the MedGuideRag demo app at http://localhost:8501 (double-click or run from any folder).
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment .venv not found - see the installation steps in README.md.
    pause
    exit /b 1
)

rem The app sends generation requests to Ollama on 127.0.0.1:11434; start it if it is not running yet.
rem findstr, not find: with Git's usr\bin on PATH, "find" resolves to the Unix tool.
tasklist /fi "imagename eq ollama.exe" /nh | findstr /i "ollama.exe" >nul
if errorlevel 1 if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"

rem Fewer threads for the numeric libraries: without this, the development machine ran out of memory.
set OMP_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
set MKL_NUM_THREADS=2
set TOKENIZERS_PARALLELISM=false

rem Listen on localhost only: by default Streamlit is reachable from the whole network.
".venv\Scripts\python.exe" -m streamlit run src\medguiderag\app\ui.py --server.address localhost --browser.gatherUsageStats false
pause
