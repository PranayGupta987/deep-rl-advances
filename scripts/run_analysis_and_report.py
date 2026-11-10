"""
Automated runner for Iteration 1 analysis and PDF report compilation.
Waits for multi_seed_results.json to be ready, then runs statistical analysis
and compiles the conference-grade PDF report.
"""

import os
import time
import subprocess
import sys


def main():
    json_path = "results/iteration1/multi_seed_results.json"
    print("[*] Waiting for multi_seed_results.json to be generated...")

    while not os.path.exists(json_path):
        time.sleep(10)
        print(".", end="", flush=True)

    print("\n[+] Results file found! Running statistical analysis & figure generation...")

    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    # Run analysis script
    res = subprocess.run([sys.executable, os.path.join(scripts_dir, "analyze_iteration1.py")], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[ERROR] analyze_iteration1.py failed:", res.stderr)
        sys.exit(1)

    print("[+] Compiling conference-grade PDF report...")
    res2 = subprocess.run([sys.executable, os.path.join(scripts_dir, "generate_pdf_report_iteration1.py")], capture_output=True, text=True)
    print(res2.stdout)
    if res2.returncode != 0:
        print("[ERROR] generate_pdf_report_iteration1.py failed:", res2.stderr)
        sys.exit(1)

    print("\n[SUCCESS] Iteration 1 analysis and PDF report generation complete!")
    print("  - Results: results/iteration1/multi_seed_results.json")
    print("  - Tables:  results/iteration1/statistical_table.tex")
    print("  - Figures: results/iteration1/*.png, *.pdf")
    print("  - Report:  results/iteration1/Iteration1_Research_Report.pdf")


if __name__ == "__main__":
    main()
