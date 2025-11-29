"""
Streamlit launcher script for the DIP Team 5 project.

This module acts as a lightweight execution wrapper for the Streamlit
application when running in Visual Studio with Conda integration as
that requires a dedicated Python startup file.

HOW TO RUN (indirectly via Visual Studio):
    Press F5 with this file set as the project Startup File.

Direct CLI equivalent:
    streamlit run app.py --server.port 8501
"""

import streamlit.web.cli as stcli
import sys

sys.argv = [
    "streamlit",
    "run",
    "app.py",
    "--server.port",
    "8501",
]

sys.exit(stcli.main())
