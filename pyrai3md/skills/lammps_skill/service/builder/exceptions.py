"""
======================================================================
Shared exceptions of the LAMMPS input builders.

`FFModelUnsupported` is raised by the force-field model reader and the
QMMM section builders alike. It lives here so `qmmm.py` and
`ff_model.py` do not import each other just for the class.

Nov 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""


class FFModelUnsupported(RuntimeError):
    """Raised when a template cannot be compressed provably.

    The caller treats this as "keep the verbatim templates", so the
    message carries the reason for the log rather than being an error.
    """
