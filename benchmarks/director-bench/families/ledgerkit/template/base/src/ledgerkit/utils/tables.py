"""Plain-text tables for reports."""


def format_table(columns, rows, align=None) -> str:
    """Render rows (sequences of str) under column headers."""
    if not rows:
        return "no data"
    cells = [list(map(str, columns))] + [list(map(str, r)) for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    align = align or ["<"] + [">"] * (len(columns) - 1)
    lines = []
    for n, row in enumerate(cells):
        lines.append("  ".join(f"{c:{a}{w}}" for c, a, w in zip(row, align, widths)).rstrip())
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
