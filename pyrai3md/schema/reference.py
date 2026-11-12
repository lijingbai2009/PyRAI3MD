"""
======================================================================
This module provides public reference data for PyRAI3MD, including
the periodic table, atomic masses, Van der Waals and covalent radii,
the pairwise single-bond length table, state labels, and physical unit
conversion constants. This is the single source of truth for elemental
data across the entire framework.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

# State label mapping: spin multiplicity index -> spectroscopic label
STATE_LABEL = {
    0: 's',
    1: 'd',
    2: 't',
    3: 'quan',
    4: 'quin',
    5: 'sext',
    6: 'sept',
    7: 'oct',
    8: 'non',
    9: 'dec',
    10: 'und',
}

# Periodic table: element symbol <-> atomic number (bidirectional)
PERIODIC_TABLE = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Lv': 115, 'Ts': 116, 'Og': 117, 'Uue': 118,
    # Reverse mapping: atomic number -> atomic number (for integer lookup)
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20,
    21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30,
    31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40,
    41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50,
    51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60,
    61: 61, 62: 62, 63: 63, 64: 64, 65: 65, 66: 66, 67: 67, 68: 68, 69: 69, 70: 70,
    71: 71, 72: 72, 73: 73, 74: 74, 75: 75, 76: 76, 77: 77, 78: 78, 79: 79, 80: 80,
    81: 81, 82: 82, 83: 83, 84: 84, 85: 85, 86: 86, 87: 87, 88: 88, 89: 89, 90: 90,
    91: 91, 92: 92, 93: 93, 94: 94, 95: 95, 96: 96, 97: 97, 98: 98, 99: 99, 100: 100,
    101: 101, 102: 102, 103: 103, 104: 104, 105: 105, 106: 106, 107: 107, 108: 108,
    109: 109, 110: 110, 111: 111, 112: 112, 113: 113, 114: 114, 115: 115, 116: 116,
    117: 117, 118: 118,
}

# Element symbols indexed by atomic number (1-based)
ELEMENT_SYMBOLS = [
    '',  # placeholder for index 0
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Lv', 'Ts', 'Og', 'Uue',
]

# Full element names indexed by atomic number (1-based)
FULL_NAMES = [
    '',
    'HYDROGEN', 'HELIUM', 'LITHIUM', 'BERYLLIUM', 'BORON', 'CARBON', 'NITROGEN', 'OXYGEN', 'FLUORINE', 'NEON',
    'SODIUM', 'MAGNESIUM', 'ALUMINUM', 'SILICON', 'PHOSPHORUS', 'SULFUR', 'CHLORINE', 'ARGON',
    'POTASSIUM', 'CALCIUM',
    'SCANDIUM', 'TITANIUM', 'VANADIUM', 'CHROMIUM', 'MANGANESE', 'IRON', 'COBALT', 'NICKEL', 'COPPER', 'ZINC',
    'GALLIUM', 'GERMANIUM', 'ARSENIC', 'SELENIUM', 'BROMINE', 'KRYPTON', 'RUBIDIUM', 'STRONTIUM',
    'YTTRIUM', 'ZIRCONIUM',
    'NIOBIUM', 'MOLYBDENUM', 'TECHNETIUM', 'RUTHENIUM', 'RHODIUM', 'PALLADIUM', 'SILVER', 'CADMIUM',
    'INDIUM', 'TIN',
    'ANTIMONY', 'TELLURIUM', 'IODINE', 'XENON', 'CESIUM', 'BARIUM', 'LANTHANUM', 'CERIUM',
    'PRASEODYMIUM', 'NEODYMIUM',
    'PROMETHIUM', 'SAMARIUM', 'EUROPIUM', 'GADOLINIUM', 'TERBIUM', 'DYSPROSIUM', 'HOLMIUM', 'ERBIUM',
    'THULIUM', 'YTTERBIUM',
    'LUTETIUM', 'HAFNIUM', 'TANTALUM', 'TUNGSTEN', 'RHENIUM', 'OSMIUM', 'IRIDIUM', 'PLATINUM', 'GOLD',
    'MERCURY',
    'THALLIUM', 'LEAD', 'BISMUTH', 'POLONIUM', 'ASTATINE', 'RADON',
    'FRANCIUM', 'RADIUM', 'ACTINIUM', 'THORIUM',
    'PROTACTINIUM', 'URANIUM', 'NEPTUNIUM', 'PLUTONIUM', 'AMERICIUM', 'CURIUM',
    'BERKELIUM', 'CALIFORNIUM', 'EINSTEINIUM', 'FERMIUM',
    'MENDELEVIUM', 'NOBELIUM', 'LAWRENCIUM',
    'RUTHERFORDIUM', 'DUBNIUM', 'SEABORGIUM', 'BOHRIUM', 'HASSIUM',
    'MEITNERIUM', 'DARMSTADTIUM', 'ROENTGENIUM', 'COPERNICIUM',
    'NIHONIUM', 'FLEROVIUM', 'MOSCOVIUM', 'LIVERMORIUM', 'TENNESSINE', 'OGANESSON',
]

# Atomic masses in g/mol, indexed by element symbol
ATOMIC_MASS = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
    'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.390,
    'Ga': 69.723, 'Ge': 72.640, 'As': 74.922, 'Se': 78.960, 'Br': 79.904,
    'Kr': 83.800, 'Rb': 85.468, 'Sr': 87.620, 'Y': 88.906, 'Zr': 91.224,
    'Nb': 92.906, 'Mo': 95.940, 'Tc': 98.000, 'Ru': 101.070, 'Rh': 102.906,
    'Pd': 106.420, 'Ag': 107.868, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.710,
    'Sb': 121.760, 'Te': 127.600, 'I': 126.905, 'Xe': 131.293, 'Cs': 132.906,
    'Ba': 137.327, 'La': 138.906, 'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.240,
    'Pm': 145.000, 'Sm': 150.360, 'Eu': 151.964, 'Gd': 157.250, 'Tb': 158.925,
    'Dy': 162.500, 'Ho': 164.930, 'Er': 167.259, 'Tm': 168.934, 'Yb': 173.040,
    'Lu': 174.967, 'Hf': 178.490, 'Ta': 180.948, 'W': 183.840, 'Re': 186.207,
    'Os': 190.230, 'Ir': 192.217, 'Pt': 195.078, 'Au': 196.967, 'Hg': 200.590,
    'Tl': 204.383, 'Pb': 207.200, 'Bi': 208.980, 'Po': 209.000, 'At': 210.000,
    'Rn': 222.000, 'Fr': 223.000, 'Ra': 226.000, 'Ac': 227.000, 'Th': 232.000,
    'Pa': 231.000, 'U': 238.000, 'Np': 237.000, 'Pu': 244.000, 'Am': 243.000,
    'Cm': 247.000, 'Bk': 247.000, 'Cf': 251.000, 'Es': 252.000, 'Fm': 257.000,
    'Md': 258.000, 'No': 259.000, 'Lr': 266.000, 'Rf': 267.000, 'Db': 268.000,
    'Sg': 269.000, 'Bh': 270.000, 'Hs': 277.000, 'Mt': 278.000, 'Ds': 281.000,
    'Rg': 282.000, 'Cn': 285.000, 'Nh': 286.000, 'Fl': 289.000, 'Lv': 293.000,
    'Ts': 294.000, 'Og': 294.000, 'Uue': 315.000,
}

# Van der Waals radii in Angstrom, indexed by element symbol
# Missing data replaced by 2.00
VDW_RADII = {
    'H': 1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53, 'B': 1.92,
    'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80,
    'S': 1.80, 'Cl': 1.75, 'Ar': 1.88, 'K': 2.75, 'Ca': 2.31,
    'Sc': 2.11, 'Ti': 2.00, 'V': 2.00, 'Cr': 2.00, 'Mn': 2.00,
    'Fe': 2.00, 'Co': 2.00, 'Ni': 1.63, 'Cu': 1.40, 'Zn': 1.39,
    'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.85,
    'Kr': 2.02, 'Rb': 3.03, 'Sr': 2.49, 'Y': 2.00, 'Zr': 2.00,
    'Nb': 2.00, 'Mo': 2.00, 'Tc': 2.00, 'Ru': 2.00, 'Rh': 2.00,
    'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58, 'In': 1.93, 'Sn': 2.17,
    'Sb': 2.00, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16, 'Cs': 3.43,
    'Ba': 2.68, 'La': 2.00, 'Ce': 2.00, 'Pr': 2.00, 'Nd': 2.00,
    'Pm': 2.00, 'Sm': 2.00, 'Eu': 2.00, 'Gd': 2.00, 'Tb': 2.00,
    'Dy': 2.00, 'Ho': 2.00, 'Er': 2.00, 'Tm': 2.00, 'Yb': 2.00,
    'Lu': 2.00, 'Hf': 2.00, 'Ta': 2.00, 'W': 1.75, 'Re': 1.66,
    'Os': 1.55, 'Ir': 1.96, 'Pt': 2.02, 'Au': 2.07, 'Hg': 1.97,
    'Tl': 2.02, 'Pb': 2.20, 'Bi': 2.00, 'Po': 2.00, 'At': 2.00,
    'Rn': 2.00, 'Fr': 2.00, 'Ra': 2.00, 'Ac': 2.00, 'Th': 2.00,
    'Pa': 2.00, 'U': 2.00, 'Np': 2.00, 'Pu': 2.00, 'Am': 2.00,
    'Cm': 2.00, 'Bk': 2.00, 'Cf': 2.00, 'Es': 2.00, 'Fm': 2.00,
    'Md': 2.00, 'No': 2.00, 'Lr': 2.00, 'Rf': 2.00, 'Db': 2.00,
    'Sg': 2.00, 'Bh': 2.00, 'Hs': 2.00, 'Mt': 2.00, 'Ds': 2.00,
    'Rg': 2.00, 'Cn': 2.00, 'Nh': 2.00, 'Fl': 2.00, 'Lv': 2.00,
    'Ts': 2.00, 'Og': 2.00, 'Uue': 2.00,
}

# Single-bond covalent radii in Angstrom, indexed by element symbol. Cordero
# et al., Dalton Trans. 2008, 2832-2838 (doi:10.1039/b801115j), through Cm.
# Where the paper resolves a radius by spin, this table carries one value:
# Mn, Fe and Co are the means of their low- and high-spin single bonds (1.50,
# 1.42, 1.38). Carbon keeps the sp3 single-bond 0.76 the geodesic port already
# used, so every H-Ar value here is the ported table's own value. Elements
# beyond Cm are absent; a consumer that needs them owns its fallback (the
# geodesic metric uses 1.5 Angstrom).
COVALENT_RADII = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84,
    'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07,
    'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
    'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.50,
    'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
    'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20,
    'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75,
    'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42,
    'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39,
    'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40, 'Cs': 2.44,
    'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
    'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94,
    'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87,
    'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70, 'W': 1.62, 'Re': 1.51,
    'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
    'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40, 'At': 1.50,
    'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06,
    'Pa': 2.00, 'U': 1.96, 'Np': 1.90, 'Pu': 1.87, 'Am': 1.80,
    'Cm': 1.69,
}

# Typical single-bond lengths in Angstrom, keyed by 'A-B'. The heavy-atom
# entries carry the single-bond distance rather than the shortest member of the
# series: perception tests `distance < bond_lib(a, b) * 1.2`, so a table keyed
# on the shortest member (C-C as the 1.20 of acetylene) puts an ordinary
# 1.53 Angstrom single bond out of reach and splits a molecule into components
# (D-28). Widening a reach cannot un-bond anything that was bonded before.
# An unlisted pair is answered by `bond_lib`'s 1.2 Angstrom default, not here.
BOND_LENGTHS = {
    'H-H': 0.74,
    'H-B': 1.19,
    'H-C': 1.09,
    'H-N': 1.02,
    'H-O': 0.96,
    'H-F': 1.03,
    'H-Cl': 1.28,
    'H-Br': 1.41,
    'C-H': 1.09,
    'C-C': 1.53,
    'C-N': 1.47,
    'C-O': 1.43,
    'C-F': 1.35,
    'C-Cl': 1.66,
    'C-Br': 1.70,
    'N-H': 1.02,
    'N-C': 1.47,
    'N-N': 1.45,
    'N-O': 1.40,
    'O-H': 0.96,
    'O-C': 1.43,
    'O-N': 1.40,
    'O-O': 1.48,
    'F-F': 1.43,
    'F-C': 1.35,
    'Cl-Cl': 1.99,
    'Cl-C': 1.66,
    'Br-Br': 2.29,
    'Br-C': 1.70,
}

# Physical unit conversion constants live in `utilities/units.py`. They are
# not re-exported here: this module stays pure data with no import statement,
# which is what lets utilities import it without a cycle. The public
# `pyrai3md.schema.<NAME>` spelling is served by `schema/__init__.py`.
