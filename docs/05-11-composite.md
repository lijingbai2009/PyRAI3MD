---
title: "5.11. COMPOSITE"
nav_order: 11
parent: "5. Keyword sections"
---

The composite section controls how the terms of a multiscale calculation are executed. The terms of one composite evaluation are the layers of an ONIOM or fragment calculation and the cross terms between them, and this section chooses whether they run one after another in this process or in a thread pool. The layers themselves are named in the control and molecular sections, not here.

The two keywords describe the mechanism and its capacity. They are the backend-term layer of the execution graph and are independent of the outer fan-out of the workflow node, so an ONIOM layer may keep its base node serial while its own terms run in a pool.

The keywords, default values, and short descriptions are listed below.

**&COMPOSITE**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| executor | serial | Chooses the execution mode of the composite terms. serial runs them one after another in this process and thread runs them in a pool. Thread mode… |
| max_workers | 1 | Sets the size of the thread pool. It is read only when executor is thread, and the value must be at least 1. The default is 1. |

Full descriptions for all available keywords are summarized below.

**executor** - Chooses the execution mode of the composite terms. serial runs them one after another in this process and thread runs them in a pool. Thread mode asks each backend whether it may run concurrently, a backend that refuses stops the run by name, and the default is serial.

**max_workers** - Sets the size of the thread pool. It is read only when executor is thread, and the value must be at least 1. The default is 1.

A thread pool that the molecular section forbids, and a worker count that disagrees with the multipool of the molecular section, are both refused before any backend directory, thread or submission exists, because the two spellings describe one capacity.
