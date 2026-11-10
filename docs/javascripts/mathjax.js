// MathJax config for pymdownx.arithmatex (generic mode).
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
    // The notation of the papers: vectors bold italic, matrices (and higher-rank arrays) bold upright sans serif.
    macros: {
      vect: ["{\\boldsymbol{#1}}", 1],
      matr: ["{\\boldsymbol{\\mathsf{#1}}}", 1],
      tensor: ["{\\boldsymbol{\\mathsf{#1}}}", 1]
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
