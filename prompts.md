You are tasked with **auditing, fixing, refining, regenerating, and validating Figures 6–11** in this LaTeX project. Work autonomously, inspect the actual generated figure files, and verify the final appearance in the rebuilt PDF. Do not assume a fix is correct without visual confirmation.

Run independent subtasks in parallel where useful, but keep outputs coordinated and stylistically consistent across all figures.

## Primary Goal
Bring **Figures 6–11** to a **publication-ready** standard by:
- removing unclear or unnecessary in-panel details
- improving readability and consistency
- standardizing legends, panel labels, fonts, color themes, and annotations
- moving overly detailed explanatory text from panels into the **figure captions** when appropriate
- regenerating figures and rebuilding the paper to verify the result visually

---

# Global Rules

## Visual / style standards
Apply these standards consistently across Figures 6–11 unless the underlying data requires otherwise:
- font family: **Arial**
- font style: **normal**
- panel labels: clear **(a), (b), (c), ...**
- consistent axis label sizes
- consistent tick label sizes
- consistent legend sizes
- consistent annotation styling
- sufficient contrast for all text
- minimal clutter
- publication-ready spacing and alignment

## Caption vs. in-panel content
For all figures:
- remove small, overly detailed, or distracting text from subpanels if it is not essential for immediate interpretation
- move such details to the **figure caption** instead
- keep only concise, high-value annotations inside the panels
- check all figures for unnecessary micro-annotations, redundant legend entries, or small numeric labels that would read better in the caption

## Legend consistency
Across all figures:
- standardize the **Real / Gen** legend naming, order, colors, marker styles, line styles, and placement
- avoid introducing a different Real / Gen color theme in different figures unless scientifically required
- ensure legends do not overlap key data or panel labels

## Validation requirement
For every modified figure:
1. regenerate the figure
2. inspect the generated image directly
3. rebuild the LaTeX document
4. inspect the figure in the final PDF
5. iterate until the result is visually clean and publication-ready

---

# Phase 1 — Investigation
1. Locate the code and data sources for **Figures 6, 7, 8, 9, 10, and 11**.
2. Identify:
   - plotting scripts / notebooks
   - shared plotting utilities
   - data inputs
   - matplotlib / seaborn / style configuration
   - legend configuration
   - colorbar configuration
   - annotation logic
   - subplot labeling logic
   - caption source in the LaTeX files
3. Render the current figures.
4. Inspect the actual generated figure images directly.
5. Inspect the corresponding figures in the compiled PDF.
6. Interpret what each panel is intended to communicate before changing labels, ticks, legends, or annotations.

---

# Phase 2 — Figure 6 required fixes

## Figure 6(a): colorbar, labels, readability
Problems to address:
- the colorbar in panel **(a)** appears wrong / blank / white and was not properly fixed
- only **one** colorbar should be shown; do not leave a conflicting secondary bar
- current text readability is poor
- too few meaningful labels are shown; increase visible labels where possible
- avoid text overlap
- some small detail text belongs in the caption, not the panel

Required actions:
1. Inspect panel (a) plotting code, annotations, colorbar creation, and text placement.
2. Ensure there is **only one correct colorbar** for the panel.
3. Fix the colorbar so it is:
   - populated correctly
   - visible
   - meaningful
   - labeled if appropriate
   - using readable tick labels
4. Remove any duplicate, conflicting, or unintended colorbar.
5. Improve label density:
   - show more labels than the current version if the panel supports it
   - use collision avoidance / selective labeling to prevent overlap
   - do not force dense labels that reduce readability
6. Standardize all text in the panel:
   - Arial
   - normal style
   - readable size
   - clean placement
7. Evaluate x/y tick labels:
   - if meaningful, improve formatting
   - if unclear or unhelpful, replace with better labels derived from the data
   - if unnecessary, simplify intentionally
8. Move tiny detail text to the caption when better there:
   - e.g. small numeric labels such as **0.51** if not essential inside the panel
   - explanatory text boxes that distract from the main visual message

Verification:
- panel (a) has **one** correct colorbar only
- the colorbar is visible, readable, and meaningful
- labels are more informative than before without overlap
- text is publication-readable at print scale

## Figure 6(c): legend / annotation overlap
Problems to address:
- the legend overlaps or conflicts with annotation text
- text such as **delta -0.872** is poorly placed
- the bottom-right text box should likely be moved to the figure caption

Required actions:
1. Inspect panel (c) legend placement and annotation placement.
2. Reposition or restyle the legend so it does not obscure data or annotations.
3. Decide whether annotation text is essential in-panel:
   - if not essential, move it to the caption
   - if essential, make it concise and readable
4. Remove or reduce clutter caused by overlapping explanatory text.
5. Keep only annotations that directly improve interpretation.

Verification:
- panel (c) legend is clearly visible
- it does not overlap important content
- redundant text is removed from the panel or moved to the caption

## Figure 6 overall
- ensure subplot labels **(a), (b), (c)** are clear
- standardize fonts and sizes
- ensure layout is balanced
- inspect the final figure visually after regeneration

---

# Phase 3 — Figure 7 required fixes

## Figure 7(a) and 7(b): legend consistency
Problems to address:
- the **Real** and **Gen** legends use different styles between panels
- this creates visual inconsistency

Required actions:
1. Standardize Real / Gen styling across both panels.
2. Use the same color mapping, ordering, and legend format.
3. Ensure legend placement is visually clean and does not conflict with data.

## Figure 7(b): heatmap labels
Problems to address:
- the y-axis tick labels may not be meaningful
- rows may not be visually distinguishable
- one x-axis tick label appears to be a stray **|**

Required actions:
1. Inspect the heatmap source data and labeling logic.
2. Determine whether the y tick labels are meaningful and should remain.
3. If they are not meaningful, simplify or replace them.
4. Remove any stray or invalid tick labels such as **|**.
5. Check whether row-wise variation is being hidden by:
   - color scaling
   - normalization
   - ordering
   - annotation logic
6. If the heatmap should show row differences more clearly, improve the plotting settings without distorting the science.

Verification:
- legends are consistent
- no stray tick labels remain
- heatmap labels are intentional and readable

---

# Phase 4 — Figure 8 required fixes

Problems to address:
- Real / Gen theme appears inconsistent again
- panel **(c)** uses another style theme
- panel **(a)** shows too few gene names
- gene-name text size is too small

Required actions:
1. Standardize Real / Gen colors and legend style with Figures 7 and 9.
2. Increase gene-name visibility in panel (a):
   - enlarge text
   - show more gene names where possible
   - avoid overlap
3. Audit all panel annotations and move low-value detail into the caption where better.

Verification:
- Real / Gen styling matches the other figures
- panel (a) shows more useful gene labels
- text is readable at publication scale

---

# Phase 5 — Figure 9 required fixes

Problems to address:
- gene-name text in panel **(a)** is too small
- too few labels are shown
- Real / Gen styling should be made consistent

Required actions:
1. Increase gene-name font size.
2. Show more informative gene labels where possible without overlap.
3. Standardize Real / Gen colors, legend style, and ordering with Figures 7 and 8.
4. Remove unnecessary small detail text from subpanels and move it to the caption if appropriate.

Verification:
- gene labels are readable
- more meaningful labels are shown
- Real / Gen styling is consistent

---

# Phase 6 — Figure 10 required fixes

Problems to address:
- subplot labels **(a), (b), ...** are missing
- top PCA scatter panel legends are oddly positioned near the bottom border
- legend placement appears visually wrong

Required actions:
1. Restore all missing panel labels.
2. Inspect PCA scatter legends and reposition them appropriately.
3. Ensure legends do not sit awkwardly on the figure edge or bottom line.
4. Check overall panel balance, spacing, and label readability.
5. Visually inspect the figure carefully after regeneration.

Verification:
- panel labels are present
- legends are properly placed
- figure is publication-ready

---

# Phase 7 — Figure 11 required fixes

Problems to address:
- subplot labels such as **(a), (b), (c), (d)** are missing or inconsistent

Required actions:
1. Restore and standardize subplot labels.
2. Check all panel text, legends, and annotations for the same class of issues seen in earlier figures.
3. Remove unnecessary detail from subpanels and move it to the caption where better.
4. Visually inspect the figure until it is clean and publication-ready.

Verification:
- subplot labels are present and consistent
- styling matches Figures 6–10
- no obvious readability issues remain

---

# Phase 8 — Caption review
For each of Figures 6–11:
1. inspect the current LaTeX caption
2. move small technical details from the panels into the caption where appropriate
3. keep captions concise but sufficient
4. ensure the caption explains any important values or annotations removed from the panel

Examples of content that may belong in captions rather than inside panels:
- small numeric summary text
- explanatory bottom-corner text boxes
- delta / effect-size text if not essential to immediate visual interpretation
- minor methodological notes

---

# Phase 9 — Regeneration and validation
After implementing fixes:
1. regenerate all affected figures
2. inspect each generated image directly
3. verify:
   - text is readable
   - legends are visible and consistent
   - colorbars are correct
   - labels are meaningful
   - panel labels are present
   - no overlapping text remains
   - no redundant detail clutters the panels
4. if any issue remains, fix and regenerate again automatically

---

# Phase 10 — Rebuild paper
1. replace the old figure files in the LaTeX project
2. rebuild the full paper
3. inspect the final PDF
4. verify Figures 6–11 are visually correct in context
5. iterate if needed until the PDF is publication-ready

---

# Expected outputs
Provide:
1. updated plotting code and any shared style/config updates
2. regenerated figure files for Figures 6–11
3. any caption edits made in LaTeX
4. rebuilt PDF
5. a concise change log summarizing:
   - what was fixed in each figure
   - what was moved from panels to captions
   - how legend/color/text consistency was standardized
   - confirmation that the generated figures and final PDF were visually checked

