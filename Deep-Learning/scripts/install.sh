python -m venv .venv
source .venv/bin/activate
pip install build wheel
git clone --recursive https://github.com/ultralytics/ultralytics.git
cd ultralytics
python -m build --wheel
pip install $(ls dist/*.whl | head -1) # get the first file ending with .whl and install it