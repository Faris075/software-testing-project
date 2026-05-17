import subprocess
import sys
import os

os.chdir(r"C:\Users\IT\software-testing-project")
sys.path.insert(0, r"C:\Users\IT\software-testing-project")

subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])
