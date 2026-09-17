# One-time owner-authorized complementary array transfer

No remote action has been performed by the author. Root/operator reviews and deploys `transfer.py` outside frozen releases. Ordinary `bstop`, `bsub`, `bresume` are subject to site authorization; never bypass DTU esub. Existing admin-only `bmod` is not used.

On a login shell with LSF commands available, from the deployed script path:

```
/zhome/15/b/215295/venvs/ccbo/bin/python transfer.py start --campaign mcbo
/zhome/15/b/215295/venvs/ccbo/bin/python transfer.py start --campaign static
```

Each command derives the actual pending set at invocation, not a hard-coded range. A race into USUSP is resumed and excluded. Original2/4 slots plus complement14/12 slots produce maximum16 per campaign; scheduler capacity decides actual allocation. New controller copies retain scientific arguments and strengthen peer checks to both arrays. A dependent finalizer runs only after all complement elements end, validates every transferred unit's complete execution identity/output hashes plus source hashes, and resumes original slots for existing-run hash-only returns. Existing final scientific audit dependencies remain intact.

Persistent `launch-records/parallelism-CAMPAIGN-attempted` prevents automatic retries. Full logs/plan/controller/submission records are stored in a timestamped sibling directory. Any partial failure leaves originals suspended for operator inspection; this is intentional. Never delete the attempt marker or clear STOP to rerun blindly. A failure before supplementary submission permits manual review/resumption of suspended originals; a failure after accepted submission requires resolving that array first. If finalizer submission fails, the accepted supplementary job ID remains in plan.json; manually submit the recorded finalize command with the correct numended dependency after review. No scientific reruns are automatic.

The finalizer performs read-only hash validation itself; it never invokes optimizer code. Three local tests cover scheduler failure states, completed-versus-running/output/source drift validation without subprocess execution, and no resume on a failed supplement. These tests do not simulate all live LSF timing races; driver is fail-closed on unrecognized/intermediate scheduler states. Parent must review before execution.

## Applied handoff

On 17 September 2026, MCBO supplement 29426675 and validator 29426676 were accepted; static supplement 29426697 and validator 29426698 were accepted. Static indices 355–720 required compressed range syntax after an explicit Bad job name rejection; the narrowly scoped recovery script records this. Pending suspensions are asynchronous and were confirmed before submission. These scripts and paths document the completed handoff; do not rerun them blindly. See EXPERIMENT_STATUS.md for the live snapshot.
