#!/usr/bin/env python
"""Install the pure-Python xxhash shim when the native DLL is blocked.

Windows Smart App Control blocks the unsigned ``_xxhash.pyd`` shipped in the
``xxhash`` wheel, which makes ``import langchain_google_genai`` fail with::

    ImportError: DLL load failed while importing _xxhash:
    An Application Control policy has blocked this file.

This script detects that failure and replaces the compiled package with the
pure-Python shim in ``shim/xxhash.py``. It is a no-op on machines where the
native extension loads normally (Linux, macOS, most Windows machines).

Usage:
    python scripts/install_xxhash_shim.py          # apply if needed
    python scripts/install_xxhash_shim.py --revert # restore the native package
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIM = ROOT / "shim" / "xxhash.py"


def site_packages() -> Path:
    import sysconfig

    return Path(sysconfig.get_paths()["purelib"])


def native_works() -> tuple[bool, str]:
    """True if a real, working xxhash import is available."""
    code = (
        "import xxhash, sys;"
        "sys.stdout.write(getattr(xxhash, '__file__', '') or '')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    lines = (proc.stderr or proc.stdout).strip().splitlines()
    return False, lines[-1] if lines else "unknown import error"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revert", action="store_true", help="reinstall the native xxhash wheel")
    args = parser.parse_args()

    if args.revert:
        print("Reinstalling native xxhash…")
        return subprocess.call(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "xxhash"]
        )

    ok, detail = native_works()
    if ok:
        print(f"Native xxhash works ({detail}); nothing to do.")
        return 0

    print("Native xxhash is unavailable:")
    print(f"  {detail}")
    print("Installing the pure-Python shim…")

    sp = site_packages()
    target = sp / "xxhash.py"
    if not SHIM.exists():
        print(f"ERROR: shim not found at {SHIM}")
        return 1

    # Remove the compiled package so the module file can take its place.
    package_dir = sp / "xxhash"
    if package_dir.exists():
        shutil.rmtree(package_dir, ignore_errors=True)
        print(f"  removed {package_dir}")
    subprocess.call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "xxhash"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    shutil.copyfile(SHIM, target)
    print(f"  wrote {target}")

    ok, detail = native_works()
    if ok:
        print(f"\nDone. xxhash now resolves to {detail}")
        return 0
    print(f"\nStill failing: {detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
