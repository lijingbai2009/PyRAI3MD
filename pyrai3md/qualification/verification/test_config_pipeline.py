"""
======================================================================
This module locks the two halves of the config-pipeline de-duplication.

The pipeline build_config -> normalize -> validate used to construct
the Config model three times. The last construction had no work to do:
validate() only reads the configuration, and a rebuilt model was
measured to dump to exactly the input's data on all 201 shipped
examples, so it now returns the model it was handed. The two
constructions that remain each change the payload -- the first turns
typed input into a model, the second applies the normalized values.

The same stage carried class guards that were always true: validate
received the dump of a typed Config, so `isinstance(size, (int,
float))` could never be false. They were removed; the checks stayed.
`root` keeps its guard because that field is Union[int, List[int]].

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import warnings

import unittest
import yaml

from pyrai3md.qualification.verification._source_paths import (
    source_path, source_read)
from pyrai3md.schema.adapter import build_config
from pyrai3md.schema.config import Config, flatten_config_dump
from pyrai3md.schema.normalize import normalize
from pyrai3md.schema.validate import validate


class TestThePipelineBuildsTheConfigTwice(unittest.TestCase):
    """One construction per value-changing stage, none at validate."""

    def _pipeline(self, name='energy'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with open(source_path('pyrai3md/examples/%s.yaml' % name)) as f:
                raw = yaml.safe_load(f)
            built = build_config(raw)
            normalized = normalize(built)
            validated = validate(normalized)
        return built, normalized, validated

    def test_the_pipeline_constructs_the_model_twice(self):
        original = Config.model_validate.__func__
        count = [0]

        def counting(cls, data, *args, **kwargs):
            count[0] += 1
            return original(cls, data, *args, **kwargs)

        Config.model_validate = classmethod(counting)
        try:
            self._pipeline()
        finally:
            Config.model_validate = classmethod(original)
        # Once for the typed build, once for the normalized values.
        self.assertEqual(count[0], 2)

    def test_validate_returns_the_model_it_was_handed(self):
        _built, normalized, validated = self._pipeline()
        self.assertIs(validated, normalized)
        self.assertEqual(validated.model_dump(), normalized.model_dump())


class TestTheGuardedValuesAreSettled(unittest.TestCase):
    """The removed guards could not be false on a Config dump."""

    def test_every_guarded_key_is_a_number_in_the_dump(self):
        data = flatten_config_dump(Config().model_dump())
        for key in ('step', 'size', 'temperature', 'thermostat_tau',
                    'barostat_tau', 'compressibility', 'maxenergy',
                    'minenergy', 'refine', 'refine_num'):
            with self.subTest(key):
                self.assertIn(key, data)
                self.assertIsInstance(data[key], (int, float))

    def test_the_source_keeps_the_root_guard_only(self):
        text = source_read('pyrai3md/schema/validate.py')
        self.assertIn('isinstance(root, (list, tuple))', text)
        self.assertIn('isinstance(value, int)', text)
        for guarded in ('isinstance(step, int)', 'isinstance(size,',
                        'isinstance(temperature,',
                        'isinstance(compressibility,',
                        'isinstance(max_val,'):
            self.assertNotIn(guarded, text)


if __name__ == '__main__':
    unittest.main()