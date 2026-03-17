import subprocess
import sys
from pathlib import Path


def main():
    app = Path(__file__).parent / "app.py"
    sys.exit(subprocess.call(["streamlit", "run", str(app)]))
