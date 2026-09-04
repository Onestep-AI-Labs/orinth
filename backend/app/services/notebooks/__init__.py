"""Notebook storage, the managed `jupyter-server`, and run artifacts.

Split the way `datasets/` is: a service that owns the on-disk shape, a runtime
that supervises a subprocess, a proxy, and a reader for what the kernel wrote.
Nothing here imports `jupyter_server` — the runtime spawns it as a subprocess and
probes for it with `importlib.util.find_spec`, which does not execute it, so the
API process stays free of the Jupyter stack even when the extra is installed.
"""

from app.services.notebooks.service import NotebookService

__all__ = ["NotebookService"]
