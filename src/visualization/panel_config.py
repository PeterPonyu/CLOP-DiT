"""PanelConfig — tweakable layout parameters for auto-refinement."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional, Dict, Tuple, List


@dataclass
class AxisConfig:
    """Per-axis configuration."""
    max_tick_labels: int = 25
    label_rotation: int = 0
    label_fontsize: int = 10
    legend_loc: str = "best"
    legend_fontsize: int = 10
    legend_ncol: int = 1
    legend_bbox: Optional[Tuple[float, float]] = None
    show_annotations: bool = True
    max_annotations: int = 5
    annotation_fontsize: int = 10
    locator_nbins_x: Optional[int] = None
    locator_nbins_y: Optional[int] = None

    # Perceptual / semantic knobs (Passes 23-30)
    alpha: Optional[float] = None           # scatter point transparency
    max_points: Optional[int] = None        # subsample scatter above this
    palette_name: Optional[str] = None      # palette override for CVD safety
    numeric_precision: int = 4              # max decimal places in labels


@dataclass
class PanelConfig:
    """Tweakable figure layout parameters for a panel."""
    panel_id: str = ""
    figsize: Tuple[float, float] = (10.0, 7.0)
    nrows: int = 1
    ncols: int = 1
    wspace: float = 0.45
    hspace: float = 0.50
    width_ratios: Optional[List[float]] = None
    height_ratios: Optional[List[float]] = None
    suptitle_fontsize: int = 11
    suptitle_y: float = 0.96
    axes: Dict[str, AxisConfig] = field(default_factory=dict)

    # Margins
    left_margin: float = 0.10
    right_margin: float = 0.95
    top_margin: float = 0.95
    bottom_margin: float = 0.08

    # Layout-aware fields for long-label handling
    max_cols_if_long_labels: int = 2
    layout_mode: str = "auto"  # "auto", "compact", "spread", "long_labels"
    long_label_threshold: int = 20  # characters; triggers layout reduction

    def get_axis(self, name: str) -> AxisConfig:
        """Get axis config, creating default if not present."""
        if name not in self.axes:
            self.axes[name] = AxisConfig()
        return self.axes[name]

    def with_updates(self, **kwargs) -> PanelConfig:
        """Return a new PanelConfig with updated fields."""
        return replace(self, **kwargs)

    def increase_figsize(self, dw: float = 0.5, dh: float = 0.5,
                         max_w: float = 14.0, max_h: float = 10.0) -> PanelConfig:
        """Return config with increased figsize, clamped to max."""
        w, h = self.figsize
        return self.with_updates(figsize=(min(w + dw, max_w), min(h + dh, max_h)))

    def increase_spacing(self, dw: float = 0.05, dh: float = 0.05) -> PanelConfig:
        """Return config with increased wspace/hspace."""
        return self.with_updates(
            wspace=min(self.wspace + dw, 0.80),
            hspace=min(self.hspace + dh, 0.80),
        )

    def reduce_figsize(self, dw: float = 0.0, dh: float = 0.3,
                       min_w: float = 4.0, min_h: float = 3.0) -> PanelConfig:
        """Return config with reduced figsize, clamped to minimum."""
        w, h = self.figsize
        return self.with_updates(figsize=(max(w - dw, min_w), max(h - dh, min_h)))

    def reduce_spacing(self, dw: float = 0.0, dh: float = 0.05,
                       min_wspace: float = 0.10, min_hspace: float = 0.20) -> PanelConfig:
        """Return config with reduced wspace/hspace, clamped to minimum."""
        return self.with_updates(
            wspace=max(self.wspace - dw, min_wspace),
            hspace=max(self.hspace - dh, min_hspace),
        )

    def reduce_cols(self, preferred_max: int = 0) -> PanelConfig:
        """Reduce ncols when long labels make the current layout too tight.

        Returns a new config with ncols reduced and nrows adjusted to
        maintain the same total number of subplot cells.
        """
        target = preferred_max or self.max_cols_if_long_labels
        if self.ncols <= target:
            return self
        total = self.nrows * self.ncols
        new_cols = target
        new_rows = -(-total // new_cols)  # ceiling division
        return self.with_updates(ncols=new_cols, nrows=new_rows)


def choose_layout(max_label_len: int, n_subplots: int,
                  threshold: int = 20) -> Tuple[int, int]:
    """Deterministic layout choice based on label length and subplot count.

    Returns (nrows, ncols) such that labels have enough width.
    """
    if max_label_len <= threshold or n_subplots <= 2:
        # Labels are short or few subplots — keep horizontal
        return (1, n_subplots) if n_subplots <= 4 else (2, -(-n_subplots // 2))
    # Long labels — reduce columns to at most 2
    max_cols = 2
    nrows = -(-n_subplots // max_cols)
    ncols = min(n_subplots, max_cols)
    return (nrows, ncols)
