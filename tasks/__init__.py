"""
Task plugins for EEVEE.

Each task family lives under `tasks/<family>/` and provides a `task.yaml`
describing:
- which task names it supports
- where to find data splits
- which DataProcessor class to use

The pipeline discovers tasks by reading these `task.yaml` files, so adding a new
task should not require editing the framework code.
"""


