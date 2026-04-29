@echo off
REM OptiMat Alloys Launcher Script for Windows
REM Usage: launch_optimat_alloys.bat [--port PORT]

setlocal enabledelayedexpansion

echo ================================================
echo OptiMat Alloys Launcher
echo ================================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: conda not found!
    echo Please install Miniconda first.
    exit /b 1
)

REM Default port
set PORT=8000

REM Parse command line arguments
:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:args_done

echo Starting OptiMat Alloys on port %PORT%...
echo ⏳ Waiting for server to be ready...
echo.

REM Start Chainlit in headless mode (background)
start /B conda run -n optimat-alloys chainlit run run_chat.py -h --port %PORT%

REM Poll HTTP endpoint until ready
set COUNTER=0
set MAX_WAIT=30

:wait_loop
curl -s http://localhost:%PORT% >nul 2>&1
if %ERRORLEVEL% EQU 0 goto server_ready

timeout /t 1 /nobreak >nul
set /A COUNTER+=1
if %COUNTER% LSS %MAX_WAIT% goto wait_loop

REM Timeout
echo ⚠️  Server didn't start within %MAX_WAIT% seconds
echo.
echo Try manually accessing: http://localhost:%PORT%
goto end

:server_ready
echo ✅ Server is ready!
echo 📂 Opening browser...
echo.
start http://localhost:%PORT%

echo 📋 Access the application at:
echo    http://localhost:%PORT%
echo.

:end
echo Press Ctrl+C to stop the server
pause
