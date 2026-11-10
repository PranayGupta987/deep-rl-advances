"""
Automated runner for Iteration 2 analysis and PDF report compilation.
Waits for results/iteration2/multi_seed_results.json to be ready,
then runs statistical analysis and compiles the conference-grade PDF report.
"""

import os
import time
import subprocess
import sys


def main():
    json_path = "results/iteration2/multi_seed_results.json"
    print("[*] Waiting for results/iteration2/multi_seed_results.json to be generated...")

    while not os.path.exists(json_path):
        time.sleep(10)
        print(".", end="", flush=True)

    print("\n[+] Iteration 2 results found! Running statistical analysis & figure generation...")

    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    # Run analysis script
    res = subprocess.run([sys.executable, os.path.join(scripts_dir, "analyze_iteration2.py")], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[ERROR] analyze_iteration2.py failed:", res.stderr)
        sys.exit(1)

    print("[+] Compiling conference-grade PDF report for Iteration 2...")
    res2 = subprocess.run([sys.executable, os.path.join(scripts_dir, "generate_pdf_report_iteration2.py")], capture_output=True, text=True)
    print(res2.stdout)
    if res2.returncode != 0:
        print("[ERROR] generate_pdf_report_iteration2.py failed:", res2.stderr)
        sys.exit(1)

    print("\n[SUCCESS] Iteration 2 analysis and PDF report generation complete!")
    print("  - Results: results/iteration2/multi_seed_results.json")
    print("  - Tables:  results/iteration2/statistical_table_iteration2.tex")
    print("  - Figures: results/iteration2/*.png, *.pdf")
    print("  - Report:  results/iteration2/Iteration2_Research_Report.pdf")


if __name__ == "__main__":
    main()
