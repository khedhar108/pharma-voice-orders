#!/usr/bin/env python
"""
Syntax Check Script for Pharma Voice Orders

Run this before committing to ensure no Python syntax errors.
Usage: python scripts/check_syntax.py

Equivalent to TypeScript's `tsc --noEmit`
"""

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent
    
    print("🔍 Running Python Syntax Checks...")
    print("=" * 50)
    
    # 1. Built-in py_compile check
    print("\n📋 Step 1: py_compile (Syntax Check)")
    python_files = list(project_root.glob("**/*.py"))
    python_files = [f for f in python_files if ".venv" not in str(f) and "__pycache__" not in str(f)]
    
    syntax_errors = []
    for py_file in python_files:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(py_file)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                syntax_errors.append((py_file, result.stderr))
        except Exception as e:
            syntax_errors.append((py_file, str(e)))
    
    if syntax_errors:
        print("❌ Syntax errors found:")
        for file, error in syntax_errors:
            print(f"  - {file.relative_to(project_root)}")
            print(f"    {error}")
        return 1
    else:
        print(f"  ✅ {len(python_files)} files passed syntax check")
    
    # 2. Ruff linting
    print("\n📋 Step 2: Ruff Linting (Style + Errors)")
    try:
        result = subprocess.run(
            ["uv", "run", "ruff", "check", "."],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print("⚠️ Ruff found issues (see above)")
            # Don't fail on style issues, just warn
        else:
            print("  ✅ Ruff check passed")
    except FileNotFoundError:
        print("  ⚠️ Ruff not installed, skipping...")
    
    print("\n" + "=" * 50)
    print("✅ All syntax checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
