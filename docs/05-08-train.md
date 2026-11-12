---
title: "5.8. TRAIN"
nav_order: 8
parent: "5. Keyword sections"
---

The train section carries only the execution layer of jobtype train. The data file, the hyperparameters and the continuation switch belong to the skill section, so this section says how the one long call is executed and names the one preprocessing step the framework performs on its input. It exists because the fit is the longest call of a run and is therefore the call a cluster submission is usually for.

A training run reports whether the record arrived, not how well the model converged: the fit error is a reading the ranking of a grid scan compares, and this section sets nothing that changes it.

The keywords, default values, and short descriptions are listed below.

**&TRAIN**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| multitask | None | Runs the training call through a local worker pool when true and in this process when false. Unset leaves the choice to the workflow class default. |
| remotetask | None | Submits the training call to the cluster through the slurm section of the environment when true, which is what this section exists for. Unset leaves… |
| multipool | 1 | Sets the number of workers of the dispatch fan-out of this workflow node. It is a concurrency count, not a core count: the thread count of one fit… |
| permutation_map | '' | Names a legacy atom-order file that expands the training set before the fit. Every non-comment row is a permutation of the atom list, counted from… |

Full descriptions for all available keywords are summarized below.

**multitask** - Runs the training call through a local worker pool when true and in this process when false. Unset leaves the choice to the workflow class default.

**remotetask** - Submits the training call to the cluster through the slurm section of the environment when true, which is what this section exists for. Unset leaves the choice to the workflow class default.

**multipool** - Sets the number of workers of the dispatch fan-out of this workflow node. It is a concurrency count, not a core count: the thread count of one fit stays in the skill section. The default is 1.

**permutation_map** - Names a legacy atom-order file that expands the training set before the skill backend is initialized, so the skill receives frames and never a map. The file holds one row per additional permutation, whitespace separated and counted from one; every row must be a permutation of the atom list and, when symbols are present, must preserve the element order. Only the atom-indexed channels are re-indexed, the expanded frames are written as an inline dataset, and the operation is idempotent. The values No, None and an empty value disable it.
