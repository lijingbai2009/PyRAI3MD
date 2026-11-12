"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Tests of the nn_pes skill.

The TensorFlow-free contract tests always run. The numeric tests train a
network on the fly and skip with a named reason when the optional neural
network stack is absent.

"""
