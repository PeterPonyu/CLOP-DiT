"""Command-line entry points for the installed CLOP-DiT package.

Exposes `clopdit-generate` as a console script after ``pip install -e .``.

The actual inference logic lives in ``scripts/inference/05_inference.py``
for compatibility with the existing pipeline orchestration. This module
locates that script relative to the package root and delegates to its
``main()`` function so that users can invoke the pipeline either as

    python scripts/inference/05_inference.py --prompt "..." ...

or, after installation, as

    clopdit-generate --prompt "..." ...

Both paths accept the same argparse arguments.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_inference_script():
    """Locate and import the inference script as a module."""
    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "inference"
        / "05_inference.py"
    )
    if not script_path.exists():
        raise FileNotFoundError(
            f"CLOP-DiT inference script not found at {script_path}. "
            "This entry point requires an editable install (pip install -e .) "
            "from a cloned CLOP-DiT repository so that the scripts/ directory "
            "is resolvable relative to the installed package."
        )
    spec = importlib.util.spec_from_file_location("_clopdit_inference", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load inference script from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_main() -> int:
    """Entry point for the ``clopdit-generate`` console script.

    Returns the exit code from the underlying inference script's ``main()``.
    """
    try:
        module = _load_inference_script()
    except (FileNotFoundError, ImportError) as exc:
        print(f"[clopdit-generate] {exc}", file=sys.stderr)
        return 1

    if not hasattr(module, "main"):
        print(
            "[clopdit-generate] inference script does not expose a main() function.",
            file=sys.stderr,
        )
        return 1

    result = module.main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(generate_main())
