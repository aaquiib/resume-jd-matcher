@echo off
REM ============================================================
REM  setup.bat — AI Resume Matcher project setup script
REM  Run this script from the project root directory.
REM ============================================================

echo.
echo ============================================================
echo   AI Resume-Job Description Matcher -- Project Setup
echo ============================================================
echo.

REM Activate virtual environment
echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment.
    echo Make sure venv\ exists in the project directory.
    pause
    exit /b 1
)
echo Done.
echo.

REM Install dependencies
echo [2/4] Installing Python dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Check requirements.txt and your internet connection.
    pause
    exit /b 1
)
echo Done.
echo.

REM Download spaCy model
echo [3/4] Downloading spaCy language model (en_core_web_sm)...
python -m spacy download en_core_web_sm
if errorlevel 1 (
    echo ERROR: spaCy model download failed.
    pause
    exit /b 1
)
echo Done.
echo.

REM Create .env file if it doesn't exist
echo [4/4] Setting up environment file...
if not exist .env (
    copy .env.example .env >nul 2>&1
    echo .env file created from .env.example
    echo IMPORTANT: Edit .env and add your OpenAI API key if you want AI-powered advice.
) else (
    echo .env file already exists. Skipping.
)
echo.

echo ============================================================
echo   Setup complete! Run the app with:
echo.
echo     streamlit run app.py
echo ============================================================
echo.
pause
