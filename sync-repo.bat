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

REM Check for any changes (modified, untracked, deleted files)
git add -A 2>nul
git diff-index --quiet --cached HEAD 2>nul
if errorlevel 1 (
    echo Found changes. Committing them...
    git commit -m "Auto-commit: %date% %time%"
) else (
    echo No uncommitted changes found.
)

REM Get current branch
for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set BRANCH=%%i
echo Current branch: %BRANCH%
echo.

REM Check if remote branch exists
git rev-parse --verify origin/%BRANCH% >nul 2>&1
if errorlevel 1 (
    echo Remote branch does not exist yet. Pushing to create it...
    git push -u origin %BRANCH%
    echo.
    echo Sync complete!
    exit /b 0
)

REM Check if local is ahead, behind, or diverged
set BEHIND=0
set AHEAD=0
for /f %%i in ('git rev-list --count HEAD..origin/%BRANCH% 2^>nul') do set AHEAD=%%i
for /f %%i in ('git rev-list --count origin/%BRANCH%..HEAD 2^>nul') do set BEHIND=%%i

echo Commits to pull: %AHEAD%
echo Commits to push: %BEHIND%
echo.

if %BEHIND% gtr 0 if %AHEAD% gtr 0 (
    echo Repository has diverged. Pulling with rebase...
    git pull --rebase origin %BRANCH%
    if errorlevel 1 (
        echo Error: Rebase failed. Please resolve conflicts manually.
        exit /b 1
    )
    echo Pushing changes...
    git push origin %BRANCH%
) else if %AHEAD% gtr 0 (
    echo Pulling changes from remote...
    git pull origin %BRANCH%
) else if %BEHIND% gtr 0 (
    echo Pushing changes to remote...
    git push origin %BRANCH%
) else (
    echo Repository is up to date!
)

echo.
echo Sync complete!
