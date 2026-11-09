"""
start_dashboard.py - Robust launcher with mandatory pre-flight port cleanup.
Guarantees stale processes on port 8501 are killed before starting the new dashboard.
"""

import os
import sys
import time
import socket
import subprocess

def is_port_in_use(port=8501):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def kill_stale_processes(port=8501):
    """Force-terminates any process currently holding the target port."""
    try:
        cmd = f"lsof -ti :{port} | xargs kill -9 2>/dev/null"
        subprocess.run(cmd, shell=True)
        time.sleep(1.2)
        print(f"🧹 Port {port} cleared of any stale processes.")
    except Exception as e:
        print(f"⚠️ Port cleanup notice: {e}")

def main():
    port = 8501
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_file = os.path.join(base_dir, "final output 3.py")
    
    if not os.path.exists(dashboard_file):
        print(f"❌ Error: Cannot find dashboard file at {dashboard_file}")
        return

    # Always kill any existing process holding port 8501
    kill_stale_processes(port)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        dashboard_file,
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    
    print(f"🚀 Launching Current Task Dashboard:")
    print(f"   Target: {dashboard_file}")
    print(f"   Command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
