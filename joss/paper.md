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

where $\Omega\subset\mathbb{R}^d$ is the domain and $\partial\Omega$ the boundaries of the latter. In particular, $\pmb{z}$ indicates the spatio-temporal coordinates vector, $\pmb{u}$ the unknown field, $\alpha$ the physical parameters, $\pmb{f}$ the forcing term, and $\mathcal{F}$ the differential operator. In addition, $\mathcal{B}$ identifies the operator indicating arbitrary initial or boundary conditions and $\pmb{g}$ the boundary function. The PINN's objective is to find a solution to the problem, which is done by approximating the true solution $\pmb{u}$ with a neural network $\hat{\pmb{u}}_\theta : \Omega \rightarrow \mathbb{R}$, with $\theta$ that defines the parameters of the network. Such a model is trained to find the optimal parameters $\theta^*$ whose minimizing the physical loss function depending on the physical conditions $\mathcal{L}_\mathcal{F}$, boundary conditions $\mathcal{L}_\mathcal{B}$ and, if available, real data $\mathcal{L}_{\text{data}}$:

The PINNs framework is completely general and applicable to different types of ordinary differential equations (ODEs), or partial differential equations (PDEs). Nevertheless, the loss function strictly depends on the problem chosen to be solved, since different operators or boundary conditions lead to different losses, increasing the difficulty to write a general and portable code for different problems. \textbf{PINA}, \emph{Physics-Informed Neural networks for Advance modeling}, is a Python library built using PyTorch that provides a user-friendly API to formalize a large variety of physical problems and solve it using PINNs easily

# Description
PINA is an open-source Python library that provides an intuitive interface for the approximated resolution of Ordinary Differential Equations and Partial Differential Equations using  a deep learning paradigm, in particular via PINNs.
The gain of popularity for PINNs in recent years, and the evolution of open-source frameworks, such as TensorFlow, Keras, and PyTorch, led to the development of several libraries, whose focus is the exploitation of PINNs to approximately solve ODEs and PDEs.
We here mention some PyTorch-based libraries, \verb+NeuroDiffEq+ [@chen2020neurodiffeq], \verb+IDRLNet+ [@peng2021idrlnet], and some TensorFlow-based libraries, such as \verb+DeepXDE+ [@lu2021deepxde], \verb+TensorDiffEq+ [@mcclenny2021tensordiffeq], \verb+SciANN+ [@haghighat2021sciann] (which is both TensorFlow and Keras-based), \verb+PyDEns+~[@koryagin2019pydens], \verb+Elvet+ [@araz2021elvet], \verb+NVIDIA SimNet+ [@hennigh2021nvidia].
Among all these frameworks, PINA emerges for its easiness of usage, allowing the users to quickly formulate the problem at hand and solve it. We have decided to build it on top of PyTorch in order to exploit the \verb+autograd+ module, as well as all the other features implemented in this framework. The final outcome is then a library with incremental complexity, capable of being used by the new users to perform the first investigation using PINNs, but also as a core framework to actively develop new features to improve the discussed methodology.

The high-level structure of the package is illustrated in Figure \ref{API_visual}; the approximated solution of a differential equation can be implemented using PINA in a few lines of code thanks to the intuitive and user-friendly interface.
Besides the user-friendly interface, PINA also offers several examples and tutorials, aiming to guide new users toward an easy exploration of the software features. The online documentation is released at \url{https://mathlab.github.io/PINA/}, while the robustness of the package is continuously monitored by unit tests. 

The API visualization in Figure \ref{API_visual} shows that a complete workflow in PINA is characterized by 3 main steps: the problem formulation, the model definition, i.e. the structure of the neural network used, and the PINN training, eventually followed by the data visualization.
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.7\textwidth]{images/API_color.pdf}
    \caption{High-level structure of the library.}
    \label{API_visual}
\end{figure}

## Problem definition
The first step is the formalization of the problem. 
The problem definition in the PINA framework is inherited from one or more problem classes (at the moment the available classes are \verb+SpatialProblem+, \verb+TimeDependentProblem+, \verb+ParametricProblem+), depending on the nature of the problem treated.
As reported in Figure \ref{API_visual}, the user has to include in the problem formulation the following components:
\begin{itemize}
    \item the information about the domain, i.e. the spatial and temporal variables, the parameters of the problem (if any), with the corresponding range of variation;
    \item the output variables, i.e. the unknowns of the problem;
    \item the conditions that the neural network has to satisfy, i.e. the differential equations, the boundary and initial conditions.
\end{itemize}
We highlight that in PINA we abandoned the classical division between physical loss, boundary loss, and data loss: all these terms are encapsulated within the \verb+Condition+ class, in order to keep the framework as general as possible. The users can indeed define all the constraints the unknown field needs to satisfy, avoiding any forced structure in the formulation and allowing them to mix heterogeneous constraints --- e.g. data values, differential boundary conditions.

## Model definition
The second fundamental step is the definition of the model of the neural network employed to find the approximated solution to the differential problem in question.
In PINA, the user has the possibility to use either a custom \verb+torch+ network model and translate it to a PINA model (with the class \verb+Network+), or to exploit one of the built-in models such as \verb+FeedForward+, \verb+MultiFeedForward+ and \verb+DeepONet+, defining their characteristics during instantiation --- i.e. number of layers, number of neurons, activation functions. The list of the built-in models will be extended in the next release of the library.

# Illustrative example
An example of the PINA workflow is the approximated resolution of a two-dimensional Poisson problem with homogeneous Dirichlet boundary conditions\footnote{which can be found online at \url{https://mathlab.github.io/PINA/_rst/tutorial2/tutorial.html}}. We consider a square domain $\Omega=[0,1]^2$ with edges $\Gamma_i$, $i=1,\dots,4$. The starting problem is the following analytical problem:
\begin{equation}
\begin{cases}
    \Delta u=\sin{(\pi x)} \sin{(\pi y)} \text{ in } \Omega, \\
    u=0 \text{ on } \Gamma_i, \, i=1,\dots, 4.
\end{cases}
\label{poisson}
\end{equation}
The exact solution to this problem is $u^{\text{exact}}(x, y)=-\dfrac{\sin{(\pi x)} \sin{(\pi y)}}{2\pi^2}$.
In this case, the \verb+Problem+ in the PINA language is inherited from the class \verb+SpatialProblem+ and incapsulates the Eq.~\eqref{poisson} in the conditions.
In the \verb+PINN+ formulation, $20 \times 20$ sampling points are generated within the domain $\Omega$ with an equispaced grid, while $20$ equispaced sampling points are considered along the edges $\Gamma_i$, $i=1,\dots, 4$.
Finally, three different types of \verb+Model+s are considered:
\begin{itemize}
    \item[(a)] A standard \verb+FeedForward+ neural network model;
    \item[(b)] A \verb+FeedForward+ network that takes the extra-feature $\mathcal{F}(x, y)=\sin{(\pi x)} \sin{(\pi y)}$ as additional input;
    \item[(c)] A \verb+FeedForward+ network that takes as additional input a \emph{learnable} extra-feature $\mathcal{G}(x, y, \alpha, \beta)=\beta \sin{(\alpha \pi x)} \sin{(\alpha \pi y)}$, where parameters $\alpha$ and $\beta$ are learned during the NN training.
\end{itemize}
Each \verb+Model+ is composed of two hidden layers of $10$ neurons each and the non-linearity is represented by the \verb+Softplus+ activation function. Each network is trained considering a learning rate of $\num{6e-3}$ for $1000$ epochs with Adam optimizer [@adam].

Figure \ref{fig:pinn_results} represents the approximated solution, the exact solution, and the absolute error on the sampling points of the domain for the three cases taken into account. Figure \ref{fig:pinn_results} and the behavior of the NN losses in Figure \ref{fig:loss} show that the extra-features improve the PINN results of one (case \ref{feat}) and six (case \ref{learnable}) orders of magnitude with respect to the standard model

# Acknowledgements

We thank our colleagues and research partners who contributed in the
former and current developments of PINA library.
This work was partially funded by European Union Funding for Research and Innovation — Horizon 2020 Program — in the framework of European Research Council Executive Agency: H2020 ERC CoG 2015 AROMA-CFD project 681447 “Advanced Reduced Order Methods with Applications in Computational Fluid Dynamics” P.I. Professor Gianluigi Rozza.

# References
