import sys
import os
import subprocess

from QtFusion.path import abs_path


def run_script(script_path):
    """
    Run the given script with the current Python environment.

    Args:
        script_path (str): Path to the script to run

    Returns:
        None
    """
    # Path to the current Python interpreter
    python_path = sys.executable

    # Build the run command
    command = f'"{python_path}" -m streamlit run "{script_path}"'

    # Execute the command
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print("Script failed to run.")


# Instantiate and run the app
if __name__ == "__main__":
    # Path to your Streamlit script
    script_path = abs_path("web.py")

    # Run the script
    run_script(script_path)
