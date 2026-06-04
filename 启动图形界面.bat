@echo off
cd /d "%~dp0"
if exist ".venv-ml\Scripts\python.exe" (
  ".venv-ml\Scripts\python.exe" -m film_mask_automation.gui.app
) else (
  python -m film_mask_automation.gui.app
)
pause
