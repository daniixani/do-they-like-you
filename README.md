Quick setup

This project needs numpy (and optionally matplotlib for plots).

PowerShell (using the configured Python at C:/msys64/ucrt64/bin/python3.11.exe):

C:/msys64/ucrt64/bin/python3.11.exe -m pip install -r requirements.txt

If you prefer conda (recommended on Windows for binary packages):

conda create -n mcdate python=3.11 numpy matplotlib
conda activate mcdate

If pip installation fails with long build times or errors, use

- a CPython distribution (regular Windows installer from python.org)
- or install via conda which provides pre-built wheels on Windows.
