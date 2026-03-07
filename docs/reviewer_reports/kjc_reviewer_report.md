# Reviewer Report for "CLOP-DiT: Text-Guided Single-Cell Gene Expression Generation via Contrastive Language-Omics Pretraining and Diffusion Transformers"

## 1. Results Evaluation
- **Strengths:** The evaluation framework is comprehensive, evaluating the model across KNN classification accuracy, steering accuracy, diversity ratio, and downstream biological concordance. The ablation study (unconditional generation collapsing to random chance) effectively demonstrates that the text conditioning drives cell-type specificity. The discussion of the high-fidelity vs. high-diversity regimes provides good practical insights for downstream applications.
- **Concerns / Discrepancies:** 
  - **CRITICAL:** There is a significant numerical discrepancy between the Abstract and the Conclusions regarding downstream biological validation metrics.
    - **Abstract:** Claims "classifier transfer (logistic regression accuracy 51.1%)" and "differential expression concordance (logFC Pearson $r = 0.17$)". 
    - **Conclusions:** Claims "classifier transfer reaches 30.8% accuracy (macro F1 0.247)" and "mean logFC Pearson $r \approx 0.39$". 
    - *Action Required:* The authors must reconcile these numbers to ensure consistency throughout the manuscript. If these represent different metrics or subsets, it must be explicitly clarified.
  - **Out-of-Distribution (OOD) Evaluation:** The authors acknowledge that OOD evaluation is missing/incomplete. Given that text-guided generation's main appeal is generating novel or rare cell states, the lack of robust OOD evaluation weakens the broader claims of the paper.

## 2. Visualization Policies & Figure Presentations
- **Strengths:** The figures are logically organized to follow the narrative (Architecture $\rightarrow$ Training $\rightarrow$ Latent Space $\rightarrow$ Core Metrics $\rightarrow$ Gene-level fidelity $\rightarrow$ Downstream validation). 
- **Concerns:**
  - **In-Panel Legends vs. Caption Descriptions:** In Figure 5 (Marker gene comparison), the authors note: "the lineage colour mapping is described in the caption rather than repeated as an in-panel legend." Similarly, Figure 6 mentions "without boxed legend frames." Relying entirely on captions for color mappings makes the figures difficult to interpret as self-contained visual units. 
    - *Action Required:* It is highly recommended to include direct, concise in-panel legends for color mappings and shapes.
  - **Missing Labels for Readability:** In Figure 4 (Per-type fidelity), the authors state: "Specific outlier type names are omitted from the bar panels for readability and are described in the surrounding text." This forces the reader to search the text to understand the figure.
    - *Action Required:* If there are too many labels, consider highlighting only the top 3 and bottom 3 outliers directly on the plot, or using a horizontal bar chart to allow for longer label names.
  - **Fréchet Distance (FD):** The authors correctly identify that FD can be misleading for conditional generation. The inclusion of Table 1 and Figure 8 to contextualize FD against conditional-task metrics is a good visual policy.

## 3. Article Writing and Structure
- **Strengths:** The manuscript is well-structured, clearly written, and provides a reproducible pipeline narrative. The separation of Methods into dataset, model architecture, and evaluation framework is logical.
- **Concerns:**
  - **Grammar/Syntax:** In the Introduction, "Section~\ref{sec:dataset}--\ref{sec:evaluation} describe..." should be pluralized to "Sections~\ref{sec:dataset}--\ref{sec:evaluation} describe...".
  - **Clarity on Limitations:** While limitations are discussed in Section 4, the statement "CLOP-DiT therefore supports a viable---but not yet fully indistinguishable---text-conditioned generation paradigm" in the conclusion is slightly wordy. It could be tightened to be more direct about the discriminator AUC (0.656) showing that generated cells are still partially distinguishable from real ones.

## Summary of Actionable Revisions
1. **Fix Numerical Discrepancies:** Reconcile the logistic regression accuracy and logFC Pearson $r$ values between the Abstract and Conclusions.
2. **Improve Figure Self-Containment:** Add in-panel color/shape legends to Figures 5 and 6, and label key outliers in Figure 4 directly on the plot.
3. **Typographical Corrections:** Fix minor grammatical errors (e.g., "Section" to "Sections" when referring to multiple sections).
