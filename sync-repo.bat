@echo off
setlocal enabledelayedexpansion

REM Change to the script's directory (the repository)
cd /d "%~dp0"

REM Check if directory exists and is a git repository
if not exist ".git" (
    echo Error: Not a git repository
    exit /b 1
)

echo Syncing Spring semester repository...
echo.

REM Fetch latest changes from remote
echo Fetching from remote...
git fetch origin 2>nul

REM Check for uncommitted or untracked changes
git status --porcelain 2>nul | findstr /r "." >nul
if %ERRORLEVEL% equ 0 (
    echo Found uncommitted or untracked changes. Committing them...
    git add -A
    git commit -m "Auto-commit: %date% %time%"
)

REM Get current branch
for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set BRANCH=%%i
echo Current branch: %BRANCH%
echo.

REM Check if remote branch exists
git rev-parse --verify origin/%BRANCH% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Remote branch does not exist yet. Pushing to create it...
    git push -u origin %BRANCH%
    echo.
    echo Sync complete!
    exit /b 0
)

REM Check if local is ahead, behind, or diverged
set BEHIND=0
set AHEAD=0
for /f %%i in ('git rev-list --count HEAD..origin/%BRANCH% 2^>nul') do set BEHIND=%%i
for /f %%i in ('git rev-list --count origin/%BRANCH%..HEAD 2^>nul') do set AHEAD=%%i

echo Local commits ahead: %AHEAD%
echo Remote commits behind: %BEHIND%
echo.

if %AHEAD% gtr 0 if %BEHIND% gtr 0 (
    echo Repository has diverged. Pulling with rebase...
    git pull --rebase origin %BRANCH%
    if %ERRORLEVEL% neq 0 (
        echo Error: Rebase failed. Please resolve conflicts manually.
        exit /b 1
    )
    echo Pushing changes...
    git push origin %BRANCH%
) else if %BEHIND% gtr 0 (
    echo Pulling changes from remote...
    git pull origin %BRANCH%
) else if %AHEAD% gtr 0 (
    echo Pushing changes to remote...
    git push origin %BRANCH%
) else (
    echo Repository is up to date!
)

echo.
echo Sync complete!
