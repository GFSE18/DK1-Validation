@echo off
setlocal
chcp 65001 >nul
set "TS20_PY=C:\Users\Foots\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "TS20_DIR=%~dp0"
if exist "%TS20_PY%" goto run_tests
echo Bundled Python was not found at:
echo %TS20_PY%
pause
exit /b 1
:run_tests
pushd "%TS20_DIR%"
"%TS20_PY%" -X utf8 launcher.py
set "TS20_EXIT=%ERRORLEVEL%"
popd
if not "%TS20_EXIT%"=="0" pause
exit /b %TS20_EXIT%
