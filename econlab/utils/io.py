from __future__ import annotations
import pandas as pd
from typing import Optional

def to_latex_table(df: pd.DataFrame, path: Optional[str] = None) -> str:
    """Export DataFrame to LaTeX table."""
    latex = df.to_latex(index=False, escape=False)
    if path:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(latex)
    return latex


def to_word_table(df: pd.DataFrame, path: str) -> None:
    """Export DataFrame to Word (.docx)."""
    from docx import Document
    doc = Document()
    table = doc.add_table(rows=1, cols=len(df.columns))
    hdr_cells = table.rows[0].cells
    for i, c in enumerate(df.columns):
        hdr_cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    doc.save(path)