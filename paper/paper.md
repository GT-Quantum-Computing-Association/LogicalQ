---
title: "LogicalQ: A toolkit for quantum circuit development with built-in, generalized quantum error correction."
tags:
  - Python
  - quantum computing
  - quantum error correction
authors:
  - name: Rasmit Devkota
    orcid: 
    equal-contrib: true
    affiliation: "1,2"
  - name: Ben Hagan
    equal-contrib: true
    affiliation: 3
  - name: Noah Song
    equal-contrib: true
    affiliation: 2
  - name: Zhixin Song
    equal-contrib: true
    affiliation: 1
  - name: David Lloyd George
    equal-contrib: true
    affiliation: 4
  - name: Younos Hashem
    equal-contrib: true
    affiliation: 1
  - name: Nolan Heffner
    equal-contrib: true
    affiliation: "1,2"
  - name: Fiyin Makinde
    equal-contrib: true
    affiliation: 5
  - name: Richard Yu
    equal-contrib: true
    affiliation: "2,6"
affiliations:
 - name: School of Physics, College of Science, Georgia Institute of Technology, Atlanta, GA 30332, USA
   index: 1
 - name: School of Mathematics, College of Science, Georgia Institute of Technology, Atlanta, GA 30332, USA
   index: 2
 - name: Ming Hsieh Department of Electrical and Computer Engineering, Viterbi School of Engineering, University of Southern California, Los Angeles, CA 90089, United States
   index: 3
 - name: Department of Physics, Duke University, Durham, North Carolina 27708, United States
   index: 4
 - name: School of Electrical and Computer Engineering, College of Engineering, Georgia Institute of Technology, Atlanta, GA 30332, USA
   index: 5
 - name: School of Computer Science, College of Computing, Georgia Institute of Technology, Atlanta, GA 30332, USA
   index: 6
date: 01 October 2025
bibliography: paper.bib
---

# Summary

Quantum computing presents a new model for computation which may significantly accelerate scientific discovery in many fields, from physics to finance. However, the current era of quantum hardware is still noisy, and error mitigation and correction techniques are necessary for executing utility-scale algorithms. Moreover, the broader scientific consensus is that quantum error mitigation and correction techniques will always be necessary to some extent. Although there are many libraries which allow researchers with some background in quantum error correction to run experiments focused on quantum error correction, there is a lack of a unified framework for general quantum error correction which is accessible to quantum computing researchers of any background and simultaneously supports diverse algorithms.

# Statement of need

`LogicalQ` is a Python toolkit for quantum circuit development with built-in, generalized quantum error correction and mitigation. `LogicalQ` is built off of and is designed to interface well with `Qiskit` package [@qiskit].

`LogicalQ` was designed to accelerate the quantum error correction workflow.
The combination of generalized quantum error correction functionality, compatibility with libraries such as Qiskit, existence of numerous demo notebooks, and overall usability will increase accessibility to quantum error corrected-research and enable deeper study into the application of quantum error correction.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from the Quantum Computing Association at Georgia Tech.

We would like to thank Jeff Young for serving as the advisor of the Quantum Computing Association at Georgia Tech and this project.

# References


