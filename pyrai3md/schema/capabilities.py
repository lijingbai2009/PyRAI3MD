"""
======================================================================
The one vocabulary of planning capabilities.

Both directions live here, and both are read from this module:

- TOKEN_TO_CAPABILITY: the jobtype token a request spells -> the
  capability it asks for;
- CAPABILITY_TO_TOKEN: the capability -> the token a skill's config is
  written with (so a skill's own profile table can answer it).

CAPABILITY_NAMES is derived from the two, plus `overlap` (a property
with no jobtype token and no profile row) -- it used to be a third,
hand-kept list that drifted from both tables (#473). `stress` and
`overlap` are capabilities without a spellable token, which is why the
name set is not just the values of the reverse table.

The module sits in the schema layer because both the data layer
(`planning/assembly.py::spell_skill_jobtype`) and the planner
(`planning/resolver.py`) read it, and the data layer cannot import the
planner without a cycle.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

# Mapping from basic jobtype token to capability name string
TOKEN_TO_CAPABILITY = {
    'energy': 'energy',
    'gradient': 'gradient',
    'nac': 'nac',
    'soc': 'soc',
    'frequency': 'hessian',
    # The analytical matrix only: the same 'hessian' capability, so a
    # multi-backend run is refused here exactly as it is for 'frequency'.
    'hessian': 'hessian',
    # The numerical route asks for no second derivative from anyone: the
    # framework differentiates the gradients it is already entitled to, so
    # these tokens stay off the 'hessian' capability and off the
    # multi-backend ban that guards it.
    'num_freq': 'gradient',
    'num_hess': 'gradient',
    'absorption': 'transition_dipole',
    'dipole': 'dipole',
    'transition': 'transition_dipole',
    # The two QMMM duties, so every token the spelling table can emit
    # reads back to the capability it came from (the round trip below
    # is the criterion). They are not user jobtype tokens; an unknown
    # token already passed through as its own capability, so listing
    # them changes no resolution, only the completeness of this table.
    'field': 'field',
    'qmmm': 'qmmm',
    # 'stress' is not a user-facing jobtype token: it is derived from
    # the ensemble (NPT) / optimize_cell switches. The mapping entry
    # keeps an explicit token resolvable if it ever reaches the planner,
    # but it is intentionally NOT in CAPABILITY_TO_TOKEN, so it never
    # alters a skill's resolve_profile.
    'stress': 'stress',
}

# Capability -> jobtype token: the spelling a skill's config receives.
# `stress` has no user-facing token and `overlap` no profile row, so
# neither appears here; both are capabilities.
CAPABILITY_TO_TOKEN = {
    'energy': 'energy',
    'gradient': 'gradient',
    'field': 'field',
    'qmmm': 'qmmm',
    'nac': 'nac',
    'soc': 'soc',
    'dipole': 'dipole',
    'transition_dipole': 'transition',
    'hessian': 'frequency',
}

# Capabilities a `hessian` request already implies; spelling them as
# tokens beside it would ask for a jobtype no profile answers.
IMPLIED_BY_HESSIAN = frozenset({'energy', 'gradient'})

# One vocabulary, both directions: every capability the two tables name
# plus `overlap`, which is neither a jobtype token nor a spellable
# capability (its request is a property with no profile row).
CAPABILITY_NAMES = frozenset(
    set(TOKEN_TO_CAPABILITY.values()) | set(CAPABILITY_TO_TOKEN)
    | {'overlap'})