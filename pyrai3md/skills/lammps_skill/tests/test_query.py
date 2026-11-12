"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

Tests for the framework-facing query interface.

Sep 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import pytest

from .. import query


class TestResolveProfile:
    """query.resolve_profile is the framework's profile-upgrade hook.

    Without it the framework keeps the schema default profile
    ('ff_energy'), so a framework-driven dynamics step (jobtype 'md')
    never requests forces and the gradient comes back all zero.
    """

    def test_hook_is_exposed(self):
        assert hasattr(query, 'resolve_profile')

    @pytest.mark.parametrize('jobtype, expected', [
        ('md', 'ff_gradient'),
        ('energy gradient', 'ff_gradient'),
        ('gradient', 'ff_gradient'),
        ('energy', 'ff_energy'),
        ('optimize', 'ff_optimize'),
    ])
    def test_jobtype_mapping(self, jobtype, expected):
        assert query.resolve_profile('ff', jobtype) == expected

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            query.resolve_profile('bogus', 'md')


class TestDefaultsCoverTheSchema:
    """Every environment knob the schema declares must be in DEFAULT.

    `planning/assembly.py` fills a skill's flat config from `query.get_defaults()` and
    refuses to write any key DEFAULT does not carry, so a field declared in
    `schema/config.py` but left out of the table is not a cosmetic gap: a
    framework-driven run drops it silently and the builder reads its own
    default instead of the value the case declared. That is exactly how a
    future `ewald_accuracy` would go missing, and no other test in this suite
    would say so.

    Four environment members stay outside the rule because the framework
    supplies them (`skill_name`, `skill_path`, `scratch_path`) or because they
    name this skill rather than configure it (`program`). The reverse
    direction is deliberately not asserted: DEFAULT also carries keys whose
    only job is to render a classic-input section (`shape`, `scale`, `factor`)
    and those bind no schema field.
    """

    FRAMEWORK_SUPPLIED = {'program', 'scratch_path', 'skill_name',
                          'skill_path'}

    def test_environment_fields_are_all_defaulted(self):
        from ..schema.config import FIELD_TO_SUBMODEL
        defaults = set(query.get_defaults())
        environment = {name for name, sub in FIELD_TO_SUBMODEL.items()
                       if sub == 'environment'} - self.FRAMEWORK_SUPPLIED
        assert environment - defaults == set()

    def test_reciprocal_accuracy_is_one_of_them(self):
        # Named witness for the case that started this check: the key has to
        # be reachable through the framework's default table, or a YAML that
        # sets it is read as if it had said nothing.
        defaults = query.get_defaults()
        assert 'ewald_accuracy' in defaults
        assert defaults['ewald_accuracy'] == '1e-8'
