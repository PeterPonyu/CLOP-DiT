import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.visualization.direct_layout import LayoutRegion, bind_figure_region


def test_bind_figure_region_sets_metadata():
    fig = plt.figure()
    region = bind_figure_region(fig, (0.10, 0.20, 0.90, 0.80))

    assert region.as_tuple() == pytest.approx((0.10, 0.20, 0.80, 0.60))
    assert fig._clop_layout_rect == pytest.approx((0.10, 0.20, 0.90, 0.80))
    assert fig._clop_layout_managed is True
    plt.close(fig)


def test_split_cols_with_explicit_gap_matches_bounds():
    region = LayoutRegion.from_bounds(0.10, 0.20, 0.90, 0.80)
    left, right = region.split_cols([1.0, 2.0], gap=0.05)

    assert left.left == pytest.approx(0.10)
    assert left.bottom == pytest.approx(0.20)
    assert left.height == pytest.approx(0.60)
    assert left.width == pytest.approx(0.25)
    assert right.left == pytest.approx(0.40)
    assert right.width == pytest.approx(0.50)
    assert right.right == pytest.approx(0.90)


def test_split_rows_with_hspace_preserves_outer_bounds():
    region = LayoutRegion.from_bounds(0.05, 0.10, 0.95, 0.90)
    top, bottom = region.split_rows([1.0, 1.0], hspace=0.30)

    assert top.top == pytest.approx(0.90)
    assert bottom.bottom == pytest.approx(0.10)
    assert top.left == pytest.approx(0.05)
    assert bottom.right == pytest.approx(0.95)
    assert top.height == pytest.approx(bottom.height)
    assert bottom.top < top.bottom


def test_grid_returns_expected_shape_and_rects():
    region = LayoutRegion.from_bounds(0.02, 0.04, 0.98, 0.96)
    grid = region.grid(2, 3, row_heights=[1.0, 1.2], col_widths=[1.0, 1.0, 0.8], hspace=0.20, wspace=0.10)

    assert len(grid) == 2
    assert all(len(row) == 3 for row in grid)
    assert grid[0][0].top == pytest.approx(0.96)
    assert grid[1][2].right == pytest.approx(0.98)
    assert grid[0][0].left < grid[0][1].left < grid[0][2].left
    assert grid[0][0].bottom > grid[1][0].top
