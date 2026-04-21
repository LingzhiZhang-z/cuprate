Give me adversarial failure cases for the current eigenstate-selection
algorithms.

Reply in Simplified Chinese.

Focus on cases where:

- occupation-based selection fails,
- energy-based selection picks the wrong manifold,
- greedy optimization can get trapped in a local minimum,
- multi-restart can still optimize the wrong objective,
- adiabatic continuity preserves the wrong branch,
- block-wise selection and global selection disagree,
- fit residual stays high even when `T11` looks acceptable.

For each case, provide:

- the failure mode,
- the code path most likely responsible,
- the smallest reproducible test,
- the smallest experiment or diagnostic that would confirm the issue.

Do not give a broad brainstorm.
Give only concrete, high-value adversarial cases.
