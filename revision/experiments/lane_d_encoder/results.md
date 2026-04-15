# Lane D — Encoder-side robustness evidence (supplementary)

**Reviewer comments:** R2.2 (design justification), R2.3 (memorization
vs generalization), R2.5 (CLOP validation).
**Retraining:** Analysis-only in the completed pass; no new CLOP+DiT
training.
**Positioning:** Supplementary robustness — used to identify whether
the dominant bottleneck is on the encoder-side representation.

The original plan in `README.md` proposed a full BiomedBERT /
PubMedBERT / BioLinkBERT text-encoder sweep. The completed revision
work instead answered the reviewer-facing uncertainty through an
encoder-side bottleneck comparison using existing cell encoders
(`scGPT-human`, `scGPT-pancancer`, and PCA), which is the evidence
actually available in this branch.

## Question

Is the dominant bottleneck caused by the learned encoder-side
representation, or downstream in the CLOP / DiT stack?

## Status

- [x] Representative encoder-side comparison completed
- [x] scGPT-human, scGPT-pancancer, and PCA evaluated on 8 datasets
- [x] Within-type compression and tightness metrics aggregated
- [x] Verdict committed: scGPT-specific compression confirmed
- [ ] Full text-encoder swap experiment not run in this revision pass

## Results

Completed evidence from
`revision/experiments/encoder_comparison/encoder_comparison_summary.json`:

| encoder | within-L2 compression ratio (median) | tightness ratio (median) |
|---|---:|---:|
| scGPT-human | 0.144 | 0.647 |
| scGPT-pancancer | 0.066 | 0.670 |
| PCA | 0.899 | 0.900 |

Reading:

1. Both scGPT variants compress within-type structure aggressively,
   while PCA in the same 512-d space preserves it almost intact.
2. The compression is therefore not a generic dimensionality effect;
   it is a learned-representation effect.
3. `scGPT-pancancer` compresses even harder than `scGPT-human`,
   showing that training-corpus diversity modulates severity but does
   not remove the bottleneck.
4. This supports the revision-wide mechanistic claim that the
   heterogeneity gap starts upstream of DiT, in the encoder-side
   representation geometry.

## Rebuttal-ready sentence

Encoder-side robustness analysis shows that the within-type
heterogeneity gap is not a generic property of 512-d embeddings, but a
property of the learned scGPT representation: PCA preserves within-type
distance almost intact (median ratio 0.90), whereas `scGPT-human` and
`scGPT-pancancer` compress it to 0.14 and 0.07 respectively. This
localises a major part of the bottleneck upstream of CLOP and DiT,
supporting our revised interpretation that the current system is better
at preserving coarse cell identity than fine within-type structure.
