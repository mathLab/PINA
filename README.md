<!--
Copyright Contributors to the Pyro project.

SPDX-License-Identifier: Apache-2.0
-->

<div align="center">
  <p>
  <img src="https://img.shields.io/badge/PyTorch%20ecosystem-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=555">
  <img src="https://img.shields.io/pypi/dm/pina-mathlab?label=downloads&logo=pypi&style=for-the-badge"
           alt="PyPI downloads"/>
  <img src="https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FmathLab%2FPINA%2Frefs%2Fheads%2F0.3%2Fpyproject.toml&style=for-the-badge&logo=python&logoColor=white"><br>
  <img src="https://img.shields.io/github/v/release/mathlab/pina?sort=date&display_name=release&style=for-the-badge">
  <img src="https://img.shields.io/github/check-runs/mathlab/pina/master?style=for-the-badge&logo=githubactions&logoColor=white&label=master">
  <img src="https://img.shields.io/github/check-runs/mathlab/pina/dev?style=for-the-badge&logo=githubactions&logoColor=white&label=dev">


  </p>
  <a href="https://github.com/mathLab/PINA">
    <img src="https://raw.githubusercontent.com/mathLab/PINA/readme/readme/pina.svg"
         alt="PINA"
         width="60%"
         style="max-width: 220px; height: auto;" />
  </a>

  <p style="margin-top: 10px; font-size: 1.1rem;">
    <b>A Unified Framework for Scientific Machine Learning</b>
  </p>
  

           
  <p style="max-width: 800px; font-size: 1rem; line-height: 1.5;">
    <b>PINA</b> is an open-source Python library designed to simplify and accelerate the development of
    <b>Scientific Machine Learning</b> (SciML) solutions, including <b>PINNs</b>, Neural Operators,
    data-driven modeling, and more.
  </p>

  <P>
  <h4 style="margin-top: 15px; margin-bottom: 0;">  Built on top of </h4>

    
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/lightning-792ee5?style=for-the-badge&logo=lightning&logoColor=white">
  <img src="https://img.shields.io/badge/pyg-1f87e6?style=for-the-badge&logo=pyg&logoColor=white">
  </P>
</div>
<hr/>


<h2>News & Announcements</h2>

<div style="border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin: 12px 0;">
  <ul style="margin: 0; padding-left: 18px; line-height: 1.55;">
    <li>
      <b>[YYYY-MM-DD]</b> – Short announcement headline.
      <a href="LINK">More</a>
    </li>
    <li>
      <b>[YYYY-MM-DD]</b> – Another update: new release / tutorial / paper / feature.
      <a href="LINK">Details</a>
    </li>
    <li>
      <b>[YYYY-MM-DD]</b> – Maintenance note / deprecation / API change.
      <a href="LINK">Read</a>
    </li>
  </ul>
</div>

<p style="margin-top: 6px;">
  <i>Want the full history?</i>
  See the <a href="https://github.com/mathLab/PINA/releases"><b>Releases</b></a> page and the
  <a href="https://github.com/mathLab/PINA/blob/master/CHANGELOG.md"><b>Changelog</b></a> (if present).
</p>

<hr/>

<h2>What's PINA</h2>

PINA provides an intuitive framework for defining, experimenting with, and solving complex problems using Neural Networks, Physics-Informed Neural Networks (PINNs), Neural Operators, and more.

- **Modular Architecture**: Designed with modularity in mind and relying on powerful yet composable abstractions, PINA allows users to easily plug, replace, or extend components, making experimentation and customization straightforward.

- **Scalable Performance**: With native support for multi-device training, PINA handles large datasets efficiently, offering performance close to hand-crafted implementations with minimal overhead.

- **Highly Flexible**: Whether you're looking for full automation or granular control, PINA adapts to your workflow. High-level abstractions simplify model definition, while expert users can dive deep to fine-tune every aspect of the training and inference process.



<img src="pina2 (1).gif">



<hr/>

<details>
<summary>
  <h2>Installation</h2>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img align="center" src="https://img.shields.io/pypi/v/pina-mathlab?style=for-the-badge&logo=pypi&logoColor=white">
</summary>

<h3>Install a stable release</h3>

<pre><code>pip install "pina-mathlab"</code></pre>

<h3>Install from source</h3>

<pre><code>git clone https://github.com/mathLab/PINA
cd PINA
git checkout master
pip install .
</code></pre>

<summary>Install with extra dependencies</summary>

<p>
To install additional packages required for development, tests, docs, or tutorials:
</p>

<pre><code>pip install "pina-mathlab[extras]"</code></pre>

<p>Available extras:</p>

<ul>
  <li><code>dev</code> for development purposes</li>
  <li><code>test</code> for running tests locally</li>
  <li><code>doc</code> for building documentation locally</li>
  <li><code>tutorial</code> for running tutorials</li>
</ul>

<hr/>
</details>


<details>
<summary>
  <h2>PINA for Dummies </h2>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img align="center" src="https://img.shields.io/badge/documentation-fa9900?style=for-the-badge&logo=readthedocs&labelColor=555">
</summary>

<p>
Solving a differential problem in <b>PINA</b> follows a clean four-step pipeline:
</p>

<ol>
  <li>
    Define the problem and constraints using the
    <a href="https://mathlab.github.io/PINA/_rst/_code.html#problems"><b>Problem API</b></a>.
  </li>
  <li>
    Design your model using PyTorch, PyTorch Geometric, or import from the
    <a href="https://mathlab.github.io/PINA/_rst/_code.html#models"><b>Model API</b></a>.
  </li>
  <li>
    Select or build a Solver using the
    <a href="https://mathlab.github.io/PINA/_rst/_code.html#solvers"><b>Solver API</b></a>.
  </li>
  <li>
    Train with the
    <a href="https://mathlab.github.io/PINA/_rst/trainer.html"><b>Trainer API</b></a>,
    powered by PyTorch Lightning.
  </li>
</ol>

```mermaid
flowchart LR
    STEP1["<h2>Problem and Data</h2> Define the mathematical problem<br>Identify constraints or import data"]
    STEP2["<h2>Model Design</h2> Build a PyTorch module Choose or customize a model"]
    STEP3["<h2>Solver Selection</h2> Use available solvers or define your own strategy"]
    STEP4["<h2>Training</h2> Optimize the model with PyTorch Lightning"]

    STEP1 e1@--> STEP2
    STEP2 e2@--> STEP3
    STEP3 e3@--> STEP4
    e1@{ animate: true }
    e2@{ animate: true }
    e3@{ animate: true }
```

<p>
Want to dive deeper? Check out the official
<a href="https://github.com/mathLab/PINA/tree/master/tutorials#pina-tutorials"><b>Tutorials</b></a>.
</p>

<hr/>
</details>

<details>
<summary>
  <h2>PINA by Examples</h2>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img align="center" src="https://img.shields.io/badge/tutorial-fa9900?style=for-the-badge">
</summary>

<details>
<summary><h3>Data-Driven Modeling Example</h3></summary>

```python  
import torch
from pina import Trainer
from pina.model import FeedForward
from pina.solver import SupervisedSolver
from pina.problem.zoo import SupervisedProblem

input_tensor  = torch.rand((10, 1))
target_tensor = input_tensor.pow(3)

# Step 1. Define problem
problem = SupervisedProblem(input_tensor, target_tensor)

# Step 2. Define model
model = FeedForward(input_dimensions=1, output_dimensions=1, layers=[64, 64])

# Step 3. Define solver
solver = SupervisedSolver(problem, model, use_lt=False)

# Step 4. Train
trainer = Trainer(solver, max_epochs=1000, accelerator="gpu")
trainer.train()
```
<hr/>

</details>

<details>
  
<summary><h3>Physics-Informed Example</h3></summary>

<p>
Consider the following differential problem:
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large\begin{cases}\frac{d}{dx}u(x)=u(x)\quad&x\in(0,1)\\u(0)=1\end{cases}"
       alt="ODE equation" />
</p>

<p>
In PINA, this can be implemented as:
</p>

```python
from pina import Trainer, Condition
from pina.problem import SpatialProblem
from pina.operator import grad
from pina.solver import PINN
from pina.model import FeedForward
from pina.domain import CartesianDomain
from pina.equation import Equation, FixedValue

def ode_equation(input_, output_):
    u_x = grad(output_, input_, components=["u"], d=["x"])
    u = output_.extract(["u"])
    return u_x - u

class SimpleODE(SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})
    domains = {
        "x0": CartesianDomain({"x": 0.0}),
        "D": CartesianDomain({"x": [0, 1]}),
    }
    conditions = {
        "bound_cond": Condition(domain="x0", equation=FixedValue(1.0)),
        "phys_cond": Condition(domain="D", equation=Equation(ode_equation)),
    }

# Step 1. Define problem
problem = SimpleODE()
problem.discretise_domain(n=100, mode="grid", domains=["D", "x0"])

# Step 2. Define model
model = FeedForward(input_dimensions=1, output_dimensions=1, layers=[64, 64])

# Step 3. Define solver
solver = PINN(problem, model)

# Step 4. Train
trainer = Trainer(solver, max_epochs=1000, accelerator="gpu")
trainer.train()
```
<hr/>
</details>
</details>

<details>
<summary>
  <h2>Contributing & Community</h2>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img align="center" src="https://img.shields.io/github/contributors/mathlab/pina?style=for-the-badge">
</summary>
<p>
We would love to develop PINA together with the community.
A great place to start is the list of
<a href="https://github.com/mathLab/PINA/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22">
  <b>good-first-issue</b>
</a>
issues.
</p>

<p>
If you would like to contribute, please read the
<a href="CONTRIBUTING.md"><b>Contributing Guide</b></a>.
</p>

<p align="center">
  <a href="https://github.com/mathLab/PINA/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=mathLab/PINA"
         alt="Contributors"
         style="max-width: 100%; height: auto;" />
  </a>
</p>

<p align="center">
  Made with <a href="https://contrib.rocks/">contrib.rocks</a>.
</p>

<hr/>
</details>

<details>
<summary>
  <h2>Citation</h2>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img align="center" src="https://img.shields.io/badge/JOSS-10.21105/JOSS.05352-blue?logo=open-access&style=for-the-badge&logoColor=white">
</summary>
<p>
If <b>PINA</b> has been significant in your research and you would like to acknowledge it, please cite:
</p>

<pre><code>Coscia, D., Ivagnes, A., Demo, N., & Rozza, G. (2023).
Physics-Informed Neural networks for Advanced modeling.
Journal of Open Source Software, 8(87), 5352.</code></pre>

<p>Or in BibTeX format:</p>

<pre><code>@article{coscia2023physics,
  title={Physics-Informed Neural networks for Advanced modeling},
  author={Coscia, Dario and Ivagnes, Anna and Demo, Nicola and Rozza, Gianluigi},
  journal={Journal of Open Source Software},
  volume={8},
  number={87},
  pages={5352},
  year={2023}
}</code></pre>
</details>


<h2>API(NA)</h2>


Here is a high-level overview of PINA’s main modules. For full details, refer to the
<a href="https://mathlab.github.io/PINA/"><b>documentation</b></a>.
  
```mermaid
flowchart TB
    PINA["<h1>pina</h1>The basic module including `Condition`, <tt>LabelTensor</tt>, `Graph` and `Trainer` API"]

    subgraph R1[" "]
        direction LR
        PROB["<h2>pina.problem</h2> Module for defining problems via base class inheritance"]
        MODEL["<h2>pina.model</h2> Module for built-in PyTorch models full architectures"]
        SOLVER["<h2>pina.solver</h2> Module for built-in solvers and abstract interfaces"]
        CALLBACK["<h2>pina.callback</h2> Module for built-in callbacks to integrate training pipelines"]
    end

    subgraph R2[" "]
        direction LR
        DOMAIN["<h2>pina.domain</h2> Module for defining geometries and set operations"]
        BLOCK["<h2>pina.block</h2> Module for built-in PyTorch models layers only"]
        OPTIM["<h2>pina.optim</h2> Module for build or import optimizers and schedulers"]
        DATA["<h2>pina.data</h2> Module for DataModules for data processing"]
    end

    subgraph R3[" "]
        direction LR
        OPERATOR["<h2>pina.operator</h2> Module for differential operators"]
        ADAPT["<h2>pina.adaptive_function</h2> Module for PyTorch learnable activations"]
        LOSS["<h2>pina.loss</h2> Module for losses and weighting strategies"]
        CONDITION["<h2>pina.condition</h2> Module for model training constraints"]
    end

    PINA --> PROB
    PINA --> MODEL
    PINA --> SOLVER
    PINA --> CALLBACK

    PROB --> DOMAIN
    MODEL --> BLOCK
    SOLVER --> OPTIM
    CALLBACK --> DATA

    DOMAIN --> OPERATOR
    BLOCK --> ADAPT
    OPTIM --> LOSS
    DATA --> CONDITION
```
