"""Verify the project is running inside an active Python virtual environment."""

import sys


def main() -> None:
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print("OK: Virtual environment is active.")
        print(f"   Python: {sys.executable}")
    else:
        print("ERROR: No virtual environment detected.")
        print("   Activate your venv first, then run this script again.")
        print("   Windows (PowerShell): .\\venv\\Scripts\\Activate.ps1")
        sys.exit(1)


if __name__ == "__main__":
    main()
