# Table II-V corrections

- Table II is now populated with generation-stage patch rate, mean cost, and mean execution statistics from the supplied 50-task run summaries.
- Official resolution is reported only for AgentForge because a saved official SWE-bench harness report is available for that configuration: 1/50 resolved, 2.0% with exact 95% Clopper-Pearson CI [0.05%, 10.65%].
- The other Table II configurations are explicitly marked "Not graded" for resolution rather than assigning unsupported official resolution rates. ReAct produced no non-empty patches, but no saved official harness report was available.
- Table III contains prior-work contextual results only; the unsupported AgentForge placeholder row was removed.
- Table IV is populated from the supplied task-level trajectories for the four factorial arms (50 attempted tasks each). The same 22 leakage-guard aborts occur in all four arms and are included in the denominator.
- Table V is a run-status table instead of presenting fabricated resolution/paired-effect values. The supplied artifact has no official reports for the ablations, and several ablation runs contain provider balance/API failures.
- Surrounding captions/prose were adjusted only where needed to keep the manuscript consistent with the corrected tables.
