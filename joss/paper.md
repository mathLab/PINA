---
title: 'Physics-Informed Neural networks for Advance modeling'
tags:
  - python
  - deep learning
  - physics-informed neural networks
  - scientific machine learning
  - differential equations.
authors: 
  - name: Dario Coscia
    orcid: 0000-0001-8833-6833
    equal-contrib: true
    affiliation: "1"
  - name: Anna Ivagnes
    orcid: 0000-0002-2369-4493
    equal-contrib: true
    affiliation: "1"
  - name: Nicola Demo
    orcid: 0000-0003-3107-9738
    equal-contrib: true
    affiliation: "1"
  - name: Gianluigi Rozza
    orcid: 0000-0002-0810-8812
    equal-contrib: true
    affiliation: "1"
affiliations:
 - name: SISSA, International School of Advanced Studies, Via Bonomea 265, Trieste, Italy
   index: 1
date: 15 March 2023
bibliography: paper.bib
---

# Introduction
Artificial Intelligence (AI) strategies are massively emerging in several fields of academia and industrial research [@deng2014deep, @Wang_2005] due to the growing disposal of data, as well as the great improvement in computational resources. In the area of applied mathematics and simulations, AI strategies are being used to solve problems where classical methods fail [@pinns].
However, the amount of data required to analyze complex systems is often insufficient to make AI predictions reliable and robust. Physics-informed neural networks (PINNs) have been formulated in [@RAISSI2019686] to overcome the issues of missing data, by incorporating the physical knowledge into the neural network training. Thus, PINNs aim to approximate any equation by solving a minimization problem in an unsupervised learning setting, learning the unknown field in order to preserve the imposed constraints (boundaries and physical residuals). Formally, we consider the general form of a differential equation, which typically presents the most challenging issues from a numerical point of view:
\begin{equation}
\begin{split}
    \mathcal{F}(\pmb{u}(\pmb{z});\alpha)&=\pmb{f}(\pmb{z}) \quad \pmb{z} \in \Omega,\\       
    \mathcal{B}(\pmb{u}(\pmb{z}))&=\pmb{g}(\pmb{z}) \quad \pmb{z} \in \partial\Omega,   
\end{split}
\end{equation}
where $\Omega\subset\mathbb{R}^d$ is the domain and $\partial\Omega$ the boundaries of the latter. In particular, $\pmb{z}$ indicates the spatio-temporal coordinates vector, $\pmb{u}$ the unknown field, $\alpha$ the physical parameters, $\pmb{f}$ the forcing term, and $\mathcal{F}$ the differential operator. In addition, $\mathcal{B}$ identifies the operator indicating arbitrary initial or boundary conditions and $\pmb{g}$ the boundary function. The PINN's objective is to find a solution to the problem, which is done by approximating the true solution $\pmb{u}$ with a neural network $\hat{\pmb{u}}_\theta : \Omega \rightarrow \mathbb{R}$, with $\theta$ that defines the parameters of the network. Such a model is trained to find the optimal parameters $\theta^*$ whose minimizing the physical loss function depending on the physical conditions $\mathcal{L}_\mathcal{F}$, boundary conditions $\mathcal{L}_\mathcal{B}$ and, if available, real data $\mathcal{L}_{\text{data}}$:
\begin{equation}
    \theta^* = \argmin_\theta \mathcal{L} = 
    \argmin_\theta (\mathcal{L}_\mathcal{F} + \mathcal{L}_\mathcal{B} + \mathcal{L}_{\text{data}}).
\end{equation}
The PINNs framework is completely general and applicable to different types of ordinary differential equations (ODEs), or partial differential equations (PDEs). Nevertheless, the loss function strictly depends on the problem chosen to be solved, since different operators or boundary conditions lead to different losses, increasing the difficulty to write a general and portable code for different problems. \textbf{PINA}, \emph{Physics-Informed Neural networks for Advance modeling}, is a Python library built using PyTorch that provides a user-friendly API to formalize a large variety of physical problems and solve it using PINNs easily

# Description
PINA is an open-source Python library that provides an intuitive interface for the approximated resolution of Ordinary Differential Equations and Partial Differential Equations using  a deep learning paradigm, in particular via PINNs.
The gain of popularity for PINNs in recent years, and the evolution of open-source frameworks, such as TensorFlow, Keras, and PyTorch, led to the development of several libraries, whose focus is the exploitation of PINNs to approximately solve ODEs and PDEs.
We here mention some PyTorch-based libraries, \verb+NeuroDiffEq+ [@chen2020neurodiffeq], \verb+IDRLNet+ [@peng2021idrlnet], and some TensorFlow-based libraries, such as \verb+DeepXDE+ [@lu2021deepxde], \verb+TensorDiffEq+ [@mcclenny2021tensordiffeq], \verb+SciANN+ [@haghighat2021sciann] (which is both TensorFlow and Keras-based), \verb+PyDEns+ [@koryagin2019pydens], \verb+Elvet+ [@araz2021elvet], \verb+NVIDIA SimNet+ [@hennigh2021nvidia].
Among all these frameworks, PINA emerges for its easiness of usage, allowing the users to quickly formulate the problem at hand and solve it. We have decided to build it on top of PyTorch in order to exploit the \verb+autograd+ module, as well as all the other features implemented in this framework. The final outcome is then a library with incremental complexity, capable of being used by the new users to perform the first investigation using PINNs, but also as a core framework to actively develop new features to improve the discussed methodology.


# Acknowledgements

We thank our colleagues and research partners who contributed in the
former and current developments of PINA library.
This work was partially funded by European Union Funding for Research and Innovation — Horizon 2020 Program — in the framework of European Research Council Executive Agency: H2020 ERC CoG 2015 AROMA-CFD project 681447 “Advanced Reduced Order Methods with Applications in Computational Fluid Dynamics” P.I. Professor Gianluigi Rozza.

# References
