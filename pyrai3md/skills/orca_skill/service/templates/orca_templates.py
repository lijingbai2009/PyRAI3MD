"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the CP2K input file templates for each block.
Coordinates are written in Angstrom (framework unit, CP2K default).

"""


# Main keyword line: ! <functional> <basis> <job keywords>
KEYWORD_TEMPLATE = '! {keywords}'

# Parallel execution block (%pal)
PAL_TEMPLATE = """%pal
  nprocs {nproc}
end
"""

# Memory per core in MB (%maxcore)
MAXCORE_TEMPLATE = '%maxcore {memory}'

# SCF control block (%scf): MaxIter aligned with the scf_maxit config;
# Print[P_MOs] 1 always dumps the MO coefficients for inspection
SCF_TEMPLATE = """%scf
  MaxIter {scf_maxit}
  Print[P_MOs] 1
end
"""

# Geometry optimization block (%geom): MaxIter aligned with opt_maxit
GEOM_TEMPLATE = """%geom
  MaxIter {opt_maxit}
end
"""

# TDDFT block; lines are indented key-value pairs (nroots, irootlist, TDA, DoSOC)
TDDFT_TEMPLATE = """%tddft
{tddft_lines}
end
"""

# Background point charges are read from an external file: ORCA refuses an
# inline block ("Quotes expected after scanning %pointcharges"). The file
# opens with the charge count, then one "q x y z" line per charge (charge in
# atomic units, position in Angstrom). Charges taken from a file keep their
# mutual interaction out of the nuclear energy, which is what the embedding
# needs -- the classical provider owns that term.
POINTCHARGES_TEMPLATE = """%pointcharges "{filename}"
"""

POINTCHARGES_FILE_TEMPLATE = """{ncharges}
{charge_block}
"""

# Geometry block: coordinates in Angstrom
GEOMETRY_TEMPLATE = """* xyz {charge} {mult}
{coord_block}*
"""
