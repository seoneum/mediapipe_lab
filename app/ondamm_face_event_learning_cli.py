"""Command-line entry point for bounded face-proxy event learning.

The CLI accepts JSON only and never opens a camera, stores media, calls a network,
or invokes GPT.  ``--demo`` is fixed-data and therefore byte-for-byte deterministic.
All outputs are observational candidates requiring a human review and are explicitly
non-diagnostic.
"""
from __future__ import annotations

try:
    from .ondamm_face_event_learning import main
except ImportError:  # direct file invocation has no package on sys.path
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app.ondamm_face_event_learning import main


if __name__ == "__main__":
    raise SystemExit(main())
