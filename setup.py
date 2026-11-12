######################################################
#
# PyRAI3MD setup file
#
# Author Jingbai Li
# Jul 28 2026
#
######################################################

import os
from setuptools import setup, find_packages

# Read long description from README if available
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as fh:
        long_description = fh.read()
else:
    long_description = (
        'PyRAI3MD is an extensible AI-driven ab initio molecular dynamics framework '
        'designed for complex molecular and periodic systems. '
        'It introduces a modular skill-based infrastructure that '
        'enables seamless integration of quantum chemistry programs, '
        'machine learning models, and autonomous simulation workflows. '
        'PyRAI3MD provides a unified framework for developing and executing '
        'advanced molecular simulations.'
    )

setup(
    name='pyrai3md',
    version='3.0.0',
    author='Jingbai Li',
    author_email='lijingbai@zspt.edu.cn',
    description='Python Rapid with Advanced Infrastructure for Artificial Intelligence Ab Initio Molecular Dynamics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.20.0',
        'pydantic>=2.0.0',
        'pyyaml>=6.0',
        # Population plots of the benchmark calibrations
        # (qualification/benchmark/plotting.py) and the pyscf
        # spectra service; the plot helper degrades gracefully
        # when the import fails at runtime.
        'matplotlib>=3.5',
    ],
    extras_require={
        # Optional Numba acceleration of the FSSH electronic
        # propagator kernels (workflow/integrator/fssh_numba.py);
        # the NumPy reference stays runnable without it.
        'numba': ['numba>=0.56'],
        # Optional neural network potential backend
        # (skills/nn_pes_skill). Kept out of install_requires: TensorFlow
        # is a ~1.4 GB install and only this one skill needs it. The skill
        # is still discovered without it and says what is missing.
        'nn': ['tensorflow>=2.20'],
    },
    entry_points={
        'console_scripts': [
            'pyrai3md=pyrai3md.pmd:main',
            'pyrai3md_agent=pyrai3md.agent:main',
            'pyrai3md_skill=pyrai3md.skills.wrapper:main',
        ],
    },
    package_data={
        'pyrai3md': ['manifest.yaml', 'skill_guide.md',
                     'examples/*.yaml', 'examples/*.yml',
                     'examples/*.init.xyz', 'examples/*.json',
                     'examples/*.charge', 'examples/ethylene.xyz',
                     # Adaptive's pretrained ethylene ensemble: the model
                     # folder is read by name from the example's launch
                     # directory, so its files ship with the package.
                     'examples/nn_ethylene_init/*',
                     'examples/nn_ethylene_init/energy_grad/*',
                     'examples/nn_ethylene_init/energy_grad/fit_stats/*',
                     # The path family's inputs: the two band endpoints, the
                     # saddle a band is given as its intermediate guess, and the
                     # vibration file an IRC reads its imaginary mode from. A
                     # band that could not read its endpoints is not a band, so
                     # they ship beside the examples that name them - the same
                     # rule the LAMMPS topologies below follow.
                     'examples/path_*.xyz', 'examples/path_*.molden',
                     'examples/h2o.data', 'examples/h2o.in.init',
                     'examples/h2o.in.settings',
                     'examples/ethanol.data', 'examples/ethanol.in.init',
                     'examples/ethanol.in.settings',
                     # The variable-cell vehicle's TIP3P box (#355): a LAMMPS
                     # topology is read by name from the launch directory, so
                     # it ships beside the example that declares it.
                     'examples/water25.data', 'examples/water25.in.init',
                     'examples/water25.in.settings',
                     # The same box with one ethylene in it (molecule 526 of
                     # 526), the carrier of the force-field-environment crossing
                     # pair. Its `.data` is its own; the two templates are
                     # byte-identical to water25's because the 98-type force-field
                     # dictionary is shared - pinned by
                     # verification/test_multicenter_periodic.py rather than left
                     # to look like an accidental copy.
                     'examples/c2h4_water25.data',
                     'examples/c2h4_water25.in.init',
                     'examples/c2h4_water25.in.settings'],
        'pyrai3md.qualification': ['pmd_env.yaml'],
        'pyrai3md.qualification.benchmark': ['*.yaml'],
        'pyrai3md.skills.molcas_skill': ['manifest.yaml', 'skill_guide.md',
                                         'tests/*.yaml'],
        'pyrai3md.skills.openqp_skill': ['manifest.yaml', 'skill_guide.md', 'tests/*.yaml'],
        'pyrai3md.skills.bagel_skill': ['manifest.yaml', 'skill_guide.md', 'tests/*.yaml'],
        'pyrai3md.skills.pyscf_skill': ['manifest.yaml', 'skill_guide.md', 'tests/*.yaml'],
        'pyrai3md.skills.orca_skill': ['manifest.yaml', 'skill_guide.md', 'tests/*.yaml'],
        'pyrai3md.skills.cp2k_skill': ['manifest.yaml', 'skill_guide.md', 'tests/*.yaml'],
        'pyrai3md.skills.xtb_skill': ['manifest.yaml', 'skill_guide.md', 'tests/*.yaml'],
        'pyrai3md.skills.lammps_skill': ['manifest.yaml', 'skill_guide.md', 'tests/*.yaml', 'tests/h2o.data', 'tests/h2o.in.init', 'tests/h2o.in.settings',
                                        'tests/qmmm_cluster_test.data', 'tests/qmmm_cluster_test.in.init', 'tests/qmmm_cluster_test.in.settings',
                                        'tests/qmmm_cluster_energy_test.data', 'tests/qmmm_cluster_energy_test.in.init', 'tests/qmmm_cluster_energy_test.in.settings',
                                        'tests/qmmm_cluster_test_qmmm_field.esfp'],
        'pyrai3md.skills.qdm_skill': ['manifest.yaml', 'skill_guide.md', 'tests/*.yaml'],
        'pyrai3md.skills.nn_pes_skill': ['manifest.yaml', 'skill_guide.md', 'tests/*.yaml', 'tests/data/*.json'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=[
        'materials science', 'machine learning', 'molecular dynamics',
        'ab initio', 'artificial intelligence', 'quantum chemistry',
    ],
)
