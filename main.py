import sys
import subprocess

def main():
    """Entry point for the application script."""
    # Use subprocess to run streamlit command in the current environment
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
