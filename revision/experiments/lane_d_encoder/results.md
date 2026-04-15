# Lane D — Text encoder comparison (supplementary)

**Reviewer comments:** R2.2 (design justification), R2.3 (memorization
vs generalization), R2.5 (CLOP validation).
**Retraining:** CLOP + DiT per encoder variant.
**Positioning:** Supplementary robustness — not the first-priority
lane.

See `revision/experiments/lane_d_encoder/README.md` for encoder
choices and metric contract.

## Question

Is the text encoder a limiting factor? If swapping to PubMedBERT or
BioLinkBERT meaningfully changes the headline metrics, the system is
encoder-sensitive. If all encoders give within-noise results, the
bottleneck is downstream (decoder / latent interface) and R2.3's
memorization concern is largely refuted.

## Status

- [ ] BiomedBERT (baseline) confirmed from frozen baseline values
- [ ] PubMedBERT CLOP + DiT trained
- [ ] BioLinkBERT CLOP + DiT trained
- [ ] Stage-1 metrics collected per encoder
- [ ] Main generation metrics collected per encoder
- [ ] Field-ablation / semantic-sensitivity repeated per encoder
- [ ] OOD prompt robustness repeated per encoder
- [ ] Verdict committed (sensitive / insensitive)

## Results

_(fill in after the runs complete)_

## Rebuttal-ready sentence

_(2–3 sentences mapping the encoder sweep to R2.2 / R2.3 / R2.5)_
