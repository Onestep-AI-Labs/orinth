"""The `orinth` command-line interface.

Deliberately empty of imports. `app/cli/main.py` answers `--help`, `version`, and
an unknown group without importing a single command module, and a package
`__init__` that pulled anything in would defeat that before dispatch ran.
"""

__all__ = ["__version__"]

#: Reported by `orinth version` and compared against the server's `/health`.
#: Tracks `backend/pyproject.toml`'s `project.version`.
__version__ = "0.1.0"
