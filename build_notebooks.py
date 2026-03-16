#!/usr/bin/env python3
"""Convert index.qmd sections into individual Jupyter notebooks."""

import json
import re
import os

# ── Read the full QMD file ──────────────────────────────────────────────────
with open('index.qmd', 'r', encoding='utf-8') as f:
    all_lines = f.readlines()

# ── Section definitions (1-indexed line numbers) ────────────────────────────
# Each entry: (notebook_name, display_title, start_line, end_line)
# First section of each module includes the module-level header/intro lines.
SECTIONS = [
    ("MLBA_S1",  "Module 1: Python Programming Fundamentals\nSection 1.1 — Python Basics and Conditional Statements",  40,  225),
    ("MLBA_S2",  "Section 1.2 — Loops in Python",                       226,  407),
    ("MLBA_S3",  "Section 1.3 — Lists, Dictionaries, and Tuples",        408,  632),
    ("MLBA_S4",  "Section 1.4 — Introduction to NumPy and Pandas",       633,  820),
    ("MLBA_S5",  "Module 2: Exploratory Data Analysis (EDA)\nSection 2.1 — Handling Missing Data",  821,  995),
    ("MLBA_S6",  "Section 2.2 — Scaling and Normalising Data",           996, 1158),
    ("MLBA_S7",  "Section 2.3 — Identifying Key Features",              1159, 1322),
    ("MLBA_S8",  "Section 2.4 — Data Visualisation for EDA",            1323, 1485),
    ("MLBA_S9",  "Module 3: Introduction to Machine Learning\nSection 3.1 — ML Concepts and Workflow",  1486, 1657),
    ("MLBA_S10", "Section 3.2 — Simple Regression Models",              1658, 1832),
    ("MLBA_S11", "Section 3.3 — Simple Classification Models",          1833, 2011),
    ("MLBA_S12", "Section 3.4 — Decision Trees and Random Forests",     2012, 2209),
    ("MLBA_S13", "Module 4: Business Applications of Machine Learning\nSection 4.1 — ML in Marketing",  2210, 2400),
    ("MLBA_S14", "Section 4.2 — ML in Finance",                        2401, 2569),
    ("MLBA_S15", "Section 4.3 — ML in Operations",                     2570, 2746),
    ("MLBA_S16", "Section 4.4 — Building End-to-End ML Solutions",     2747, 2964),
]

SETUP_CODE = "import warnings\nwarnings.filterwarnings('ignore')"

# ── Helpers ──────────────────────────────────────────────────────────────────

def to_source(text: str) -> list:
    """Convert multi-line text to Jupyter source list (lines with \\n except last)."""
    lines = text.split('\n')
    out = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return out


def clean_markdown(text: str) -> str:
    """Remove or simplify Quarto-specific markdown elements."""
    # Remove {#sec-xxx} anchors from headers
    text = re.sub(r'\s*\{#sec-[\w-]+\}', '', text)
    # Convert ::: {.callout-note ...} title line → **bold**
    text = re.sub(
        r':::\s*\{\.callout-\w+[^}]*\}\s*\n(## (.+?)\n)?',
        lambda m: ('**' + m.group(2).strip() + '**\n\n') if m.group(2) else '',
        text
    )
    # Remove closing :::
    text = re.sub(r'^:::\s*$', '', text, flags=re.MULTILINE)
    # Remove leftover {.xxx} divs
    text = re.sub(r'^\{\.[\w-]+.*\}$', '', text, flags=re.MULTILINE)
    # Collapse 3+ blank lines → 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def parse_section_to_cells(section_text: str) -> list:
    """Parse QMD section text into [(type, content)] list."""
    cells = []
    parts = re.split(r'(```\{python\}.*?```)', section_text, flags=re.DOTALL)

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue

        if stripped.startswith('```{python}') and stripped.endswith('```'):
            # Code cell — strip fences
            code = stripped[len('```{python}'):-3]
            code = code.strip('\n')
            # Remove Quarto #| cell options
            code_lines = [ln for ln in code.split('\n') if not ln.startswith('#|')]
            code = '\n'.join(code_lines).strip('\n')
            if code.strip():
                cells.append(('code', code))
        else:
            md = clean_markdown(stripped)
            if md.strip():
                cells.append(('markdown', md))

    return cells


def make_notebook(cells: list) -> dict:
    """Build a Jupyter notebook dict from (type, content) cells."""
    nb_cells = [
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "setup-00",
            "metadata": {"tags": ["setup"]},
            "outputs": [],
            "source": to_source(SETUP_CODE),
        }
    ]

    for i, (ctype, content) in enumerate(cells):
        cid = f"cell-{i:04d}"
        if ctype == 'markdown':
            nb_cells.append({
                "cell_type": "markdown",
                "id": cid,
                "metadata": {},
                "source": to_source(content),
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "id": cid,
                "metadata": {},
                "outputs": [],
                "source": to_source(content),
            })

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0",
            },
        },
        "cells": nb_cells,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
os.makedirs('Notebooks', exist_ok=True)

for name, title, start, end in SECTIONS:
    section_lines = all_lines[start - 1: end]
    section_text = ''.join(section_lines)
    cells = parse_section_to_cells(section_text)
    nb = make_notebook(cells)

    out_path = os.path.join('Notebooks', f'{name}.ipynb')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"  {name}.ipynb  ({len(cells)} cells)")

print(f"\nAll {len(SECTIONS)} notebooks written to ./Notebooks/")
