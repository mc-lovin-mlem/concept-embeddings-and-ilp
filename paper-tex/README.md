# LaTeX sources for KI2020 paper

These are the sources to build the KI2020 paper 

*Rabold, Johannes, Gesina Schwalbe, and Ute Schmid. 2020. “Expressive Explanations of DNNs by Combining Concept Analysis with ILP.” In KI 2020: Advances in Artificial Intelligence. Lecture Notes in Computer Science. Bamberg, Germany: Springer International Publishing.*

The build is tested with a TeXLive2019 and the Springer
[llncs.cls](./llncs.cls) version 2.20 LaTeX class.
To build call:
```bash
pdflatex concept_embeddings_and_ilp.tex
bibtex concept_embeddings_and_ilp
pdflatex concept_embeddings_and_ilp.tex
pdflatex concept_embeddings_and_ilp.tex
```

The script `../script/paper_layerwise_results_graphics.py` 
contains the code to produce the graph for the concept embedding results.
