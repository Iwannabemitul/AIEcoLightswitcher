#!/usr/bin/env python3
"""
run_verification.py - Standalone Verification Test Runner for AIEcoLightswitcher.

Executes the automated 4-tier E2E testing suite headlessly, prints a formatted
colored scorecard with per-tier statistics, and exits with code 0 on complete
pass (or code 1 on failure).
"""

import sys
import os
import time
import unittest
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ANSI terminal color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_GREEN = "\033[42m\033[30m"
    BG_RED = "\033[41m\033[37m"


class TierTestResult(unittest.TestResult):
    """
    Custom TestResult collector categorizing outcomes by verification tier.
    """
    def __init__(self):
        super().__init__()
        self.test_records: List[Dict] = []
        self._test_start_time = 0.0

    def startTest(self, test):
        super().startTest(test)
        self._test_start_time = time.perf_counter()

    def addSuccess(self, test):
        super().addSuccess(test)
        elapsed = time.perf_counter() - self._test_start_time
        self._record(test, "PASS", elapsed)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        elapsed = time.perf_counter() - self._test_start_time
        self._record(test, "FAIL", elapsed, err)

    def addError(self, test, err):
        super().addError(test, err)
        elapsed = time.perf_counter() - self._test_start_time
        self._record(test, "ERROR", elapsed, err)

    def _determine_tier(self, test_name: str, class_name: str) -> str:
        if "Tier1" in class_name or "TC_T1" in test_name:
            return "Tier 1: Feature Coverage"
        elif "Tier2" in class_name or "TC_T2" in test_name:
            return "Tier 2: Boundary & Corner Cases"
        elif "Tier3" in class_name or "TC_T3" in test_name:
            return "Tier 3: Cross-Feature Combinations"
        elif "Tier4" in class_name or "TC_T4" in test_name:
            return "Tier 4: Real-World Scenarios"
        return "Tier 1: General Features"

    def _record(self, test, status: str, elapsed: float, err=None):
        method_name = test._testMethodName
        class_name = test.__class__.__name__
        doc = (test._testMethodDoc or "").strip().split("\n")[0]
        tier = self._determine_tier(method_name, class_name)
        err_msg = ""
        if err:
            err_msg = self._exc_info_to_string(err, test)

        self.test_records.append({
            "method": method_name,
            "class": class_name,
            "tier": tier,
            "doc": doc,
            "status": status,
            "elapsed_ms": elapsed * 1000.0,
            "error": err_msg
        })


def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}================================================================================
           AIEcoLightswitcher — E2E Verification & Test Suite
                    Showcase & Interview Edition
================================================================================{Colors.RESET}
{Colors.DIM}Execution Mode : 100% Headless CPU Execution (No External Hardware Required)
Simulation     : MockVideoSource + MockHardwareController
Device Target  : CPU / NVIDIA CUDA Fallback Safe
Test Framework : Python Standard Library unittest{Colors.RESET}
--------------------------------------------------------------------------------
"""
    print(banner)


def main():
    # Enable ANSI escape sequences on Windows if needed
    if os.name == "nt":
        os.system("")

    print_banner()

    start_wall_time = time.time()

    # Load test suite
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")

    # Run tests using custom collector
    result = TierTestResult()
    suite.run(result)

    total_duration = time.time() - start_wall_time

    # Group test records by tier
    tiers: Dict[str, List[Dict]] = {
        "Tier 1: Feature Coverage": [],
        "Tier 2: Boundary & Corner Cases": [],
        "Tier 3: Cross-Feature Combinations": [],
        "Tier 4: Real-World Scenarios": []
    }

    for rec in result.test_records:
        tier_name = rec["tier"]
        if tier_name not in tiers:
            tiers[tier_name] = []
        tiers[tier_name].append(rec)

    # Print individual test results by tier
    for tier_name, records in tiers.items():
        if not records:
            continue
        print(f"\n{Colors.BOLD}{Colors.WHITE}>> {tier_name}{Colors.RESET}")
        print(f"{Colors.DIM}{'-' * 80}{Colors.RESET}")
        for rec in records:
            status = rec["status"]
            if status == "PASS":
                status_str = f"{Colors.GREEN}[ PASS ]{Colors.RESET}"
            elif status == "FAIL":
                status_str = f"{Colors.RED}[ FAIL ]{Colors.RESET}"
            else:
                status_str = f"{Colors.YELLOW}[ ERRR ]{Colors.RESET}"

            method_display = rec["method"].replace("test_", "")
            ms_str = f"{rec['elapsed_ms']:>6.1f}ms"
            doc_snippet = rec["doc"] if rec["doc"] else method_display

            print(f"  {status_str}  {Colors.BOLD}{method_display:<44}{Colors.RESET} {Colors.DIM}({ms_str}){Colors.RESET}")
            if doc_snippet and doc_snippet != method_display:
                print(f"          {Colors.DIM}--> {doc_snippet}{Colors.RESET}")

    # Print failures and errors if any
    if result.failures or result.errors:
        print(f"\n{Colors.RED}{Colors.BOLD}============================ FAILURE / ERROR DETAILS ============================{Colors.RESET}")
        for rec in result.test_records:
            if rec["status"] in ("FAIL", "ERROR"):
                print(f"\n{Colors.RED}{Colors.BOLD}[{rec['status']}] {rec['class']}.{rec['method']}{Colors.RESET}")
                print(rec["error"])

    # Print Summary Scorecard
    print(f"\n{Colors.BOLD}============================= VERIFICATION SCORECARD ============================={Colors.RESET}")
    print(f"{'Tier Name':<38} | {'Total':<6} | {'Passed':<6} | {'Failed':<6} | {'Errors':<6} | {'Status'}")
    print(f"{'-' * 38}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}-+-------")

    all_passed = True
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0

    for tier_name, records in tiers.items():
        tier_total = len(records)
        tier_pass = sum(1 for r in records if r["status"] == "PASS")
        tier_fail = sum(1 for r in records if r["status"] == "FAIL")
        tier_err = sum(1 for r in records if r["status"] == "ERROR")

        total_tests += tier_total
        total_passed += tier_pass
        total_failed += tier_fail
        total_errors += tier_err

        if tier_fail > 0 or tier_err > 0:
            tier_status = f"{Colors.RED}FAILED{Colors.RESET}"
            all_passed = False
        else:
            tier_status = f"{Colors.GREEN}PASSED{Colors.RESET}"

        print(f"{tier_name:<38} | {tier_total:<6} | {tier_pass:<6} | {tier_fail:<6} | {tier_err:<6} | {tier_status}")

    print(f"{'-' * 38}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}-+-------")

    pass_pct = (total_passed / total_tests * 100.0) if total_tests > 0 else 0.0
    overall_status = f"{Colors.BG_GREEN} ALL TESTS PASSED (100%) {Colors.RESET}" if all_passed else f"{Colors.BG_RED} VERIFICATION FAILED {Colors.RESET}"

    print(f"{Colors.BOLD}{'TOTAL':<38} | {total_tests:<6} | {total_passed:<6} | {total_failed:<6} | {total_errors:<6} | {overall_status}{Colors.RESET}")
    print(f"Total Verification Time : {total_duration:.2f} seconds")
    print(f"Pass Rate               : {pass_pct:.1f}%\n")

    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}SUCCESS: All verification tiers passed flawlessly! System is test-ready for showcase.{Colors.RESET}\n")
        sys.exit(0)
    else:
        print(f"{Colors.RED}{Colors.BOLD}FAILURE: One or more verification tiers failed. Check logs above.{Colors.RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
