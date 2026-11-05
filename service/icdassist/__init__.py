"""icdassist -- self-hosted confidence-tiered ICD coding assistance.

The deployable form of the method in docs/paper_draft_v2_directionA.md: embed notes once with a
frozen encoder, fit a linear probe on the site's OWN codes, calibrate an auto-accept threshold to a
precision bar, and route each code to either auto-accept or a human coder.

Nothing in this package sends a note anywhere but the embedding endpoint the operator configures,
which is expected to be on their own network. No model weights and no probe trained on our patients
are shipped: every site fits its own, which takes about two minutes once the notes are embedded.
"""
__version__ = "0.1.0"

from .probe import LinearProbe              # noqa: F401
from .calibrate import Router               # noqa: F401
from .embed import Embedder                 # noqa: F401
