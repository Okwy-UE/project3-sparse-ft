
# project3-sparse-ft



Unstructured sparse-to-dense fine-tuning experiments (Cerebras CS-3 vs NVIDIA GPUs),

with a Phoenix-compatible baseline track.



## Layout

- configs/   : experiment configs (Cerebras + GPU)

- scripts/   : runners and utilities

- src/       : dataset hooks, eval helpers, utilities

- results/   : *summaries only* (raw runs are ignored by default)



## Run discipline

Every run produces:

- config.yaml, command.txt, env.txt, stdout.log, metrics.json

stored under a per-run directory (ignored by git by default).

