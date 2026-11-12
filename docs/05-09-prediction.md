---
title: "5.9. PREDICTION"
nav_order: 9
parent: "5. Keyword sections"
---

The prediction section carries only the execution layer of jobtype prediction. The test file, the error statistics and the raw comparison products belong to the skill section, so this section says how the one evaluation is executed and nothing else. It has the same three keywords as the execution layer of the train section.

A prediction serves a trained model folder over every frame of the test file and reports the errors of the served values. The statistics are read from the result, never from the human log.

The keywords, default values, and short descriptions are listed below.

**&PREDICTION**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| multitask | None | Runs the evaluation through a local worker pool when true and in this process when false. Unset leaves the choice to the workflow class default. |
| remotetask | None | Submits the evaluation to the cluster through the slurm section of the environment when true. Unset leaves the choice to the workflow class default. |
| multipool | 1 | Sets the number of workers of the dispatch fan-out of this workflow node. A prediction is one evaluation, so this is not where cores are spent: the… |

Full descriptions for all available keywords are summarized below.

**multitask** - Runs the evaluation through a local worker pool when true and in this process when false. Unset leaves the choice to the workflow class default.

**remotetask** - Submits the evaluation to the cluster through the slurm section of the environment when true. Unset leaves the choice to the workflow class default.

**multipool** - Sets the number of workers of the dispatch fan-out of this workflow node. A prediction is one evaluation, so this is not where cores are spent: the thread count stays in the skill section. The default is 1.
