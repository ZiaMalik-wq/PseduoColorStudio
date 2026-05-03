Color Studio - Setup and Run

Prerequisites
- Python 3.10+ installed and on PATH

Windows (PowerShell)
1) Create and activate a virtual environment
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1

2) Install dependencies
	python -m pip install --upgrade pip
	pip install -r requirements.txt

3) Run the project
	python main.py

macOS / Linux (bash/zsh)
1) Create and activate a virtual environment
	python3 -m venv .venv
	source .venv/bin/activate

2) Install dependencies
	python -m pip install --upgrade pip
	pip install -r requirements.txt

3) Run the project
	python main.py

Notes
- If you already have an active virtual environment, skip the creation step.
- Windows only: If you get a PowerShell error saying "running scripts is disabled on this system", run this command first: Set-ExecutionPolicy Unrestricted -Scope CurrentUser
