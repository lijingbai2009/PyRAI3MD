---
title: "5.6. SEARCH"
nav_order: 6
parent: "5. Keyword sections"
---

The search section configures the hyperparameter grid scan of jobtype search. Its grid maps the keywords a skill declares as tunable to the candidate values to try, and the scan is the Cartesian product of those lists. The framework never interprets the vocabulary: which keyword may vary and what a candidate means is the declaration of the skill that would train with it. The section also says how the points are executed, where their directories live, and how the finished folders are ranked.

A scan trains one model per grid point and ranks the finished folders; it does not judge them against a threshold, and only the fit error of a point is comparable across points. A rerun into the same grid_root skips the folders that are already complete and ranks them again, and retrieve reads the folders of a previous scan without submitting anything.

The keywords, default values, and short descriptions are listed below.

**&SEARCH**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| grid | {} | Maps each tunable keyword to the list of candidate values, for example depth to [1, 2] and nn_size to [8, 16]. The scan is the Cartesian product of… |
| retrieve | 0 | Ranks the folders a previous scan left under grid_root instead of submitting anything. A value of 0 runs the scan and a non-zero value reads the disk… |
| sort_by | '' | Names the error channel the points are ranked on. Empty takes the first channel the trained profile reports. The default is empty. |
| sort_stat | max | Chooses the statistic of a point: max ranks by the worst ensemble member and mean by the average over the members. A misspelled value is refused. The… |
| sort_order | asc | Chooses the order of the ranking: asc puts the smallest error first and desc the largest. The default is asc. |
| grid_root | ./grid-search | Sets the root directory of the scan. Each point gets one model directory under it, named NN-&lt;title&gt;-&lt;key&gt; after its values, so a restart re-derives… |
| max_grid | 0 | Refuses a scan that expands to more than this many points. A value of 0 states no limit. The default is 0. |
| multitask | None | Fans the points of the scan out over a local worker pool when true and runs them one at a time in this process when false. Unset leaves the choice to… |
| remotetask | None | Submits each point as a remote child runtime through the slurm section of the environment when true. Unset leaves the choice to the workflow class… |
| multipool | 1 | Sets the number of points that run at once. It is a concurrency count, not a core count: the thread and rank counts of one training stay in the skill… |

Full descriptions for all available keywords are summarized below.

**grid** - Maps each tunable keyword to the list of candidate values, and the scan is the Cartesian product of those lists. The keyword vocabulary belongs to the skill: a name the skill does not declare as searchable, a name written twice, and an entry with no candidate are all refused when the scan is planned, which is the one place that can see the declaration. The classic carrier writes the whole grid on one keyword with semicolons between the entries, and one grid line per keyword reaches the same mapping because a repeated key keeps both values. There is no start, stop and step generator: the values to try are written out.

**retrieve** - Reads the folders a previous scan left under grid_root and ranks them, submitting nothing. It is the mode to use when the trainings are already on disk and only the table is wanted, and it is the same reader the parent process falls back to for a point whose own child died, so a folder read back and a folder reported at the end of its own fit cannot disagree. A value of 0 runs the scan instead.

**sort_by** - Names the error channel the points are ranked on, for example the energy channel or a gradient channel of the trained profile. An empty value takes the first channel the profile reports, which the plan derives rather than the user guessing it. A point that reports no value for the channel is ranked last in either order, because it has no error rather than a large one. The default is empty.

**sort_stat** - Chooses the statistic of a point: max ranks by the worst ensemble member and mean by the average over the members. A misspelled value is refused. The default is max.

**sort_order** - Chooses the order of the ranking: asc puts the smallest error first and desc the largest. The default is asc.

**grid_root** - Sets the root directory of the scan. Each point gets one model directory under it, named NN-&lt;title&gt;-&lt;key&gt; after its values, so a restart re-derives the folder rather than guessing it. The default is ./grid-search.

**max_grid** - Refuses a scan that expands to more than this many points. A Cartesian product grows quietly, so the limit is a guard against submitting a scan larger than the machine or the account can carry. A value of 0 states no limit. The refusal names the size the grid expanded to and the limit the run declared.

**multitask** - Fans the points of the scan out over a local worker pool when true and runs them one at a time in this process when false. Unset leaves the choice to the workflow class default.

**remotetask** - Submits each point as a remote child runtime through the slurm section of the environment when true. Unset leaves the choice to the workflow class default.

**multipool** - Sets the number of points that run at once. It is a concurrency count, not a core count: the thread and rank counts of one training stay in the skill section. The default is 1.

A scan ranks, it never judges: there is no acceptance threshold, and a run whose points all completed is reported as completed even when a few points failed. A scan is also refused under a retraining session, because a continuation cannot change the architecture a grid point would vary.
