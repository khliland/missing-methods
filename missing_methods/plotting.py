"""Shared plotting helpers and default class colors."""

from __future__ import annotations


DEFAULT_CLASS_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def get_class_colors(n_classes: int, class_colors=None) -> list[str]:
    """Return plotting colors for ``n_classes`` categories.

    If ``class_colors`` is omitted, colors are taken from
    ``DEFAULT_CLASS_COLORS`` and repeated as needed.
    """
    n_classes = int(n_classes)
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")

    if class_colors is not None:
        colors = list(class_colors)
        if len(colors) != n_classes:
            raise ValueError("class_colors must have one color per class")
        return colors

    repeats = (n_classes + len(DEFAULT_CLASS_COLORS) - 1) // len(DEFAULT_CLASS_COLORS)
    return list((DEFAULT_CLASS_COLORS * repeats)[:n_classes])


__all__ = ["DEFAULT_CLASS_COLORS", "get_class_colors"]