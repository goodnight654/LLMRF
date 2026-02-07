# 1
conda create -n ads  python=3.12.2
conda activate ads
# 2
C:
cd C:\Program Files\Keysight\ADS2025_Update1\tools\python\wheelhouse
# 3
python -m pip install -r venv_requirements.txt --find-links .
# 4
python -m pip list