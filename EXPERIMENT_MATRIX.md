# VI-ReID parallel experiment matrix

All variants are independent source-code copies derived from
`SCM_for_VI-ReID-Code-improved`. No model weights, logs, datasets, or Git
history are included.

| Directory | Main variable | Default setting | Risk |
|---|---|---|---|
| `SCM_for_VI-ReID-Code-improved` | Fair improved baseline | invariant KL + center alignment | Low |
| `SCM_for_VI-ReID-v1-hc` | Hetero-center triplet | weight 0.50, margin 0.30 | Low |
| `SCM_for_VI-ReID-v2-circle` | Circle pair loss | weight 0.10, margin 0.25, gamma 128 | Medium |
| `SCM_for_VI-ReID-v3-mixstyle` | Cross-modal MixStyle | p 0.50, alpha 0.10 | Medium |
| `SCM_for_VI-ReID-v4-adamw-cosine` | AdamW + cosine decay | 5-epoch warmup | Low/medium |
| `SCM_for_VI-ReID-v5-combo` | HC + MixStyle + AdamW/cosine | conservative defaults above | Medium/high |

Submit each directory as an independent GPU job. Every directory is
self-contained and uses the same entry-point name. For example:

```bash
CUDA_VISIBLE_DEVICES=0 ./SCM_for_VI-ReID-v1-hc/run_variant.sh
```

The launchers preserve the original repository's evaluation behavior: evaluate
every epoch and save the checkpoint with the best Rank-1 score.

For a fair comparison, keep dataset split, trial, seed, input size, backbone,
pretraining, and evaluation protocol identical. Report mean and standard
deviation across at least three seeds for the shortlisted variants.
