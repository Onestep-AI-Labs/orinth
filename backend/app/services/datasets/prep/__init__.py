"""Prep agent: turning raw uploads into datasets that are ready to train.

`readiness` is pure and dependency-free, so it is safe to import from
``DatasetService.summary``. The detection, planning, and apply stages land here
in later stages of phase 21.
"""

from app.services.datasets.prep.readiness import readiness_for

__all__ = ["readiness_for"]
