"""MkDocs build hooks for the tutorial notebooks.

Converts notebooks to markdown at build time via nbconvert, writing directly
into docs/tutorials/<folder>/ (gitignored, generated) so they render through
the site's normal Python-Markdown + pymdownx.arithmatex pipeline, same as
every other page -- math included. mkdocs-jupyter's own HTML export bypasses
that pipeline entirely (its output isn't wrapped in the `arithmatex` class our
MathJax config restricts itself to), which is why LaTeX wasn't rendering
there. This also avoids staging a byte-for-byte copy of each source .ipynb
inside docs_dir, which the mkdocs-jupyter plugin required.

Raw HTML <img>/<video> tags authored in the notebooks (pointing at
../../docs/figs/...) pass through nbconvert unrewritten; a post-build pass
over the built HTML recomputes each page's actual depth and fixes them, since
MkDocs' own relative-link resolution only rewrites markdown-authored links,
not arbitrary raw HTML content.

The same post-build pass prepares the figures for both colour schemes: the
white background of every PNG of the built site becomes transparent, and the
<img> tags of the figures drawn on a dark background get the class `fig-dark`
(docs/stylesheets/figures.css inverts the colours in the scheme that needs it).
The source figures are left as they are, so they still read on GitHub.
"""
import re
import shutil
from pathlib import Path

import nbformat
import numpy as np
from nbconvert import MarkdownExporter
from nbconvert.writers import FilesWriter
from PIL import Image

REPO_ROOT = Path(__file__).parent
SRC = REPO_ROOT / "tutorials"
DEST = REPO_ROOT / "docs" / "tutorials"

# Topics rendered in full: every notebook of the folder.
RENDERED_TOPICS = ["0_How_to_repo", "1_Introduction_to_real-time_DA",
                   "2_Real-time_DA_Thermoacoustics", "4_Real-time_DA_ROMs"]
# Single notebooks rendered from the other topics (relative to SRC, no suffix).
RENDERED_NOTEBOOKS = [
    "3_Introduction_to_ROMs/31_POD_SPOD",
    "3_Introduction_to_ROMs/35_POD_ESN_cylinder",
]

# Matches a (possibly already-mangled) relative reference to the figs/ dir,
# with or without a literal "docs/" segment, so it self-corrects regardless
# of how many ../ segments survived the conversion.
FIGS_HREF_RE = re.compile(r'(src|href)="(?:\.\./)*(?:docs/)?figs/([^"]*)"')

IMG_TAG_RE = re.compile(r'<img\b[^>]*?\bsrc="([^"#?]+)"[^>]*>')
FIGURE_SUFFIXES = {".png", ".gif", ".jpg", ".jpeg"}
DARK_LUMINANCE = 96  # mean border level (0-255) below which a figure counts as dark


def _has_dark_background(rgba):
    """True if the border of an RGBA array, flattened on white, is dark on average."""
    border = np.concatenate([rgba[0], rgba[-1], rgba[:, 0], rgba[:, -1]]).astype(float)
    alpha = border[:, 3:] / 255.0
    return (border[:, :3] * alpha + 255.0 * (1.0 - alpha)).mean() < DARK_LUMINANCE


def _white_to_alpha(rgba):
    """Make white transparent: the result composited on white is the input."""
    rgba = rgba.astype(float) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:] + (1.0 - rgba[..., 3:])  # flatten on white
    alpha = 1.0 - rgb.min(axis=-1, keepdims=True)
    colour = np.where(alpha > 0, (rgb - (1.0 - alpha)) / np.maximum(alpha, 1e-12), 0.0)
    return np.round(np.concatenate([colour, alpha], axis=-1) * 255).astype(np.uint8)


def prepare_figures(site_dir):
    """Make the PNG backgrounds transparent; return the dark-background figures."""
    dark = set()
    for path in site_dir.rglob("*"):
        if path.suffix.lower() not in FIGURE_SUFFIXES or "assets" in path.relative_to(site_dir).parts:
            continue
        with Image.open(path) as im:
            rgba = np.asarray(im.convert("RGBA"))
        if _has_dark_background(rgba):
            dark.add(path.resolve())
        elif path.suffix.lower() == ".png":
            Image.fromarray(_white_to_alpha(rgba), "RGBA").save(path, optimize=True)
    return dark


# Bare, same-directory cross-notebook links, e.g. ](00_Class_Model.ipynb).
NOTEBOOK_LINK_RE = re.compile(r"\]\(([\w-]+)\.ipynb\)")


def on_pre_build(config, **kwargs):
    exporter = MarkdownExporter()
    nb_paths = [p for t in RENDERED_TOPICS for p in sorted((SRC / t).glob("*.ipynb"))]
    nb_paths += [SRC / f"{n}.ipynb" for n in RENDERED_NOTEBOOKS]
    rendered = {p.stem: p.parent.name for p in nb_paths}

    def fix_notebook_link(m):
        stem = m.group(1)
        # Within the rendered set: point at its .md page (possibly in another
        # topic folder). Otherwise (not rendered yet): fall back to GitHub.
        if stem in rendered:
            return f"](../{rendered[stem]}/{stem}.md)"
        nb_path = next(SRC.rglob(f"{stem}.ipynb"), None)
        folder = nb_path.parent.name if nb_path else ""
        return f"](https://github.com/andreanovoa/romda/blob/main/tutorials/{folder}/{stem}.ipynb)"

    for topic_dest in {DEST / p.parent.name for p in nb_paths}:
        shutil.rmtree(topic_dest, ignore_errors=True)
        topic_dest.mkdir(parents=True)

    for nb_path in nb_paths:
        nb = nbformat.read(nb_path, as_version=4)
        body, resources = exporter.from_notebook_node(
            nb, resources={"output_files_dir": f"{nb_path.stem}_files"}
        )
        body = NOTEBOOK_LINK_RE.sub(fix_notebook_link, body)
        FilesWriter(build_directory=str(DEST / nb_path.parent.name)).write(
            body, resources, notebook_name=nb_path.stem
        )


def on_post_build(config, **kwargs):
    site_dir = Path(config["site_dir"])
    dark = prepare_figures(site_dir)

    def tag_dark(m, html_dir):
        if (html_dir / m.group(1)).resolve() not in dark:
            return m.group(0)
        return m.group(0).replace("<img", '<img class="fig-dark"', 1)

    for html_path in site_dir.rglob("*.html"):
        depth = len(html_path.relative_to(site_dir).parent.parts)
        up = "../" * depth

        text = html_path.read_text(encoding="utf-8")
        fixed = FIGS_HREF_RE.sub(lambda m: f'{m.group(1)}="{up}figs/{m.group(2)}"', text)
        fixed = IMG_TAG_RE.sub(lambda m: tag_dark(m, html_path.parent), fixed)
        if fixed != text:
            html_path.write_text(fixed, encoding="utf-8")


# ── Google-style docstrings under the global numpy style ─────────────────────
# mkdocstrings sets the parser of a block's root object to the block's
# `docstring_style` (the global numpy one when the block sets none), and its
# members keep the parser they were loaded with. Google-style modules then render
# raw `Args:` text. This griffe extension (mkdocs.yml, `extensions`) pins the
# Google parser on every docstring with a Google section header and no numpy
# section underline, whatever the block options say.
import griffe

GOOGLE_SECTION_RE = re.compile(r"^\s*(Args|Arguments|Returns|Yields|Raises|Attributes|Examples?):\s*$", re.M)
NUMPY_UNDERLINE_RE = re.compile(r"^\s*-{3,}\s*$", re.M)


class _GoogleDocstring(griffe.Docstring):
    """A docstring whose parser is always Google; assignments are ignored."""

    parser = property(lambda self: griffe.Parser.google, lambda self, value: None)
    parser_options = property(lambda self: {}, lambda self, value: None)


class GoogleDocstrings(griffe.Extension):
    def on_instance(self, *, obj, **kwargs):
        doc = obj.docstring
        if doc is not None and GOOGLE_SECTION_RE.search(doc.value) and not NUMPY_UNDERLINE_RE.search(doc.value):
            doc.__class__ = _GoogleDocstring
