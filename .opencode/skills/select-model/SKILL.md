---
name: select-model
description: >-
  Guides users through selecting, configuring, or creating a PyTorch model
  for a PINA problem. Use when the user asks about models, architecture,
  neural network choice, or how to build/choose a model for their PINA
  problem. Also triggers on phrases like "what model should I use",
  "choose a model", "create a model", "neural network", "architecture",
  "model selection", or when the user mentions specific model names
  (FeedForward, DeepONet, FNO, PirateNet, etc.).
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: users
  workflow: model-creation
---

# Select a Model for a PINA Problem

> [!IMPORTANT]
> Read [RULES.md](../RULES.md) before using this skill — it applies to all skills.

Use this skill to choose the right neural network architecture for a PINA
problem, configure its parameters, and optionally enhance it with embeddings
or blocks.

## Step 1 — Determine input/output dimensions

> What data goes **in** and what comes **out**?

### If the user has a problem object

Extract dimensions directly from the problem:

```python
input_dim = len(problem.input_variables)
output_dim = len(problem.output_variables)
```

### If the user does **not** have a problem

Ask the user about their data structure. Be specific:

> What are the **input variables** and **output variables** of your problem?
> (e.g., input: x, y, t → output: u, v)

If the user refuses to provide this, reply:

> I cannot select a model without knowing the input/output structure of
> your data.

## Step 2 — Select model architecture

> What kind of mapping does your model need to learn?

Use the decision tree below. For each model, the table gives the class,
typical use case, and whether it works with a problem-based or manual
dimension setup.

| Model class | Input shape | Output shape | Typical use case |
|---|---|---|---|
| `FeedForward` | `(batch, input_dim)` | `(batch, output_dim)` | Standard MLP for scalar/vector PDEs, any smooth mapping |
| `ResidualFeedForward` | `(batch, input_dim)` | `(batch, output_dim)` | Deep MLP that needs stable gradient flow (Wang et al. 2021) |
| `PirateNet` | `(batch, input_dim)` | `(batch, output_dim)` | Multi-scale / high-frequency PDEs, mitigates spectral bias |
| `KolmogorovArnoldNetwork` | `(batch, input_dim)` | `(batch, output_dim)` | Learnable spline-based activations, fewer params than MLP |
| `MultiFeedForward` | `(batch, input_dim)` | `(batch, output_dim)` | Multiple independent sub-networks (subclass and define `forward()`) |
| `Spline` | `(batch, 1)` | `(batch, 1)` | Univariate B-spline curve fitting |
| `SplineSurface` | `(batch, 2)` | `(batch, 1)` | Bivariate B-spline surface fitting |
| `VectorizedSpline` | `(batch, input_dim)` | `(batch, output_dim)` | Parallel independent splines per feature |
| `DeepONet` | branch + trunk | `(batch, output_dim)` | Operator learning: map sensor data to solution field |
| `MIONet` | multiple branch + trunk | `(batch, output_dim)` | Multiple-input operator learning (advanced DeepONet) |
| `FNO` | `(batch, C, *grid)` | `(batch, C', *grid)` | Neural operator on regular grids (Fourier convolution) |
| `AveragingNeuralOperator` | field + coords | `(batch, C', *grid)` | Neural operator with local averaging |
| `LowRankNeuralOperator` | field + coords | `(batch, C', *grid)` | Neural operator with low-rank kernel approximation |
| `GraphNeuralOperator` | graph (PyG Data) | graph (PyG Data) | Neural operator on unstructured/point cloud data |
| `EquivariantGraphNeuralOperator` | graph (PyG Data) | graph (PyG Data) | 3D dynamics with rotational/translational equivariance |
| `SINDy` | `(batch, input_dim)` | `(batch, output_dim)` | Discover governing equations from data |
| Custom `nn.Module` | any | any | Advanced use: hand-crafted architecture |

### Decision tree

Ask these questions in order until a clear match emerges:

1. **Operator learning?** — mapping between function spaces (e.g., PDE
   param to solution field)?
   - Structured grid → **`FNO`** or similar
   - Unstructured/graph data → **`GraphNeuralOperator`** or
     **`EquivariantGraphNeuralOperator`** (if 3D rotational equivariance)
   - Field + coordinates (any geometry) → **`AveragingNeuralOperator`** or
     **`LowRankNeuralOperator`** or similar
   - Sensor-to-field (branch + trunk) → **`DeepONet`** / **`MIONet`** or similar

2. **Sparse regression?** — want to discover governing equations as
   symbolic expressions from data? → **`SINDy`** or similar

3. **Spline-based fitting?**
   - Single variable → **`Spline`** or similar
   - 2D surface → **`SplineSurface`** or similar
   - Independent splines per input feature → **`VectorizedSpline`** or similar

4. **KAN-style?** — spline activations on edges instead of fixed
   activation functions? → **`KolmogorovArnoldNetwork`**  or similar

5. **Standard pointwise mapping?** (most PDE problems)
   - High-frequency / multi-scale solution → **`PirateNet`**  or similar
     (has built-in Fourier Feature Embedding)
   - Very deep network, gradient flow concerns → **`ResidualFeedForward`** or similar
   - Simple/small problem, baseline → **`FeedForward`** or similar
   - Need multiple independent subnetworks → **`MultiFeedForward`** or similar
     (you subclass it and define `forward()`)

6. **None of the above?** → User should provide a custom `torch.nn.Module`.

### Constructing the model

Once the model class is selected, construct it with the correct dimensions:

```python
# Example: FeedForward
model = FeedForward(
    input_dimensions=input_dim,
    output_dimensions=output_dim,
    inner_size=64,
    n_layers=4,
    func=nn.Tanh,
)

# Example: PirateNet
model = PirateNet(
    input_dimension=input_dim,
    inner_size=64,
    output_dimension=output_dim,
    n_layers=3,
    activation=nn.Tanh,
)
```

## Step 3 — Enhancements and embeddings

> Does your problem benefit from feature transformations?

### Fourier Feature Embedding (multi-scale behaviour)

If the PDE solution oscillates at multiple frequencies (multi-scale), use
`FourierFeatureEmbedding` as a pre-processing block.

```python
from pina.model.block import FourierFeatureEmbedding

embedding = FourierFeatureEmbedding(
    input_dimension=input_dim,
    output_dimension=128,  # must be even
    sigma=2.0,             # std of random matrix; use larger sigma for
                           # higher-frequency features, or a list for
                           # multi-scale (future API support)
)

# Use with any model by passing embedding as a module
# or compose manually in a custom forward():
class EnhancedFeedForward(nn.Module):
    def __init__(self, input_dim, output_dim, inner_size):
        super().__init__()
        self.embedding = FourierFeatureEmbedding(input_dim, 128, sigma=2.0)
        self.ff = FeedForward(128, output_dim, inner_size=inner_size)
    def forward(self, x):
        return self.ff(self.embedding(x))
```

**When to use FourierFeatureEmbedding:**
- The PDE solution has features at multiple length scales
- Standard `FeedForward` fails to converge
- The problem exhibits "spectral bias" (NN learns low frequencies first)

**When NOT to use:** The solution is smooth and single-scale —
`FeedForward` works well on its own.

### Periodic Boundary Embedding

If the problem has known periodic boundary conditions, use
`PeriodicBoundaryEmbedding` on the relevant input dimensions:

```python
from pina.model.block import PeriodicBoundaryEmbedding

# For a problem with input variables x, y where x is periodic with period 2π
embedding = PeriodicBoundaryEmbedding(
    input_dimension=2,
    periods={"x": 2 * 3.14159},  # dict maps variable name → period
)

# Or if all dimensions have the same period:
embedding = PeriodicBoundaryEmbedding(
    input_dimension=2,
    periods=2 * 3.14159,
)
```

### Residual Connections

For architectures that need deeper layers without vanishing gradients, use
`ResidualBlock` as a building block:

```python
from pina.model.block import ResidualBlock

block = ResidualBlock(
    input_dim=64,
    output_dim=64,
    hidden_dim=128,
    activation=nn.ReLU(),
)
```

### Orthogonalization

To enforce orthonormal features at some layer (helps training stability):

```python
from pina.model.block import OrthogonalBlock

ortho = OrthogonalBlock(dim=-1)  # orthonormalize along last dimension
```

### Wrapping a model with an embedding

For models that do **not** have built-in embedding support (like
`FeedForward`), compose them manually as shown above, or construct a
sequential wrapper:

```python
class WrappedModel(nn.Module):
    def __init__(self, embedding, base_model):
        super().__init__()
        self.embedding = embedding
        self.base = base_model
    def forward(self, x):
        return self.base(self.embedding(x))

model = WrappedModel(embedding, FeedForward(128, output_dim, ...))
```

## Step 4 — Ask clarifying questions

If the user provides very little information, ask these questions to narrow
down the choice:

1. **What kind of data?** Regular grid / point cloud / graph / sensor data?
4. **Does the solution have multi-scale / high-frequency behaviour?**
5. **Are there known periodic boundary conditions?**
6. **Very deep network needed?** (residual connections help)
7. **Inputs are field functions + coordinates?** (operator learning)
8. **Need to discover symbolic equations from data?** (SINDy)

## Templates

### Template 1: Standard PDE with FeedForward

```python
from pina.model import FeedForward

model = FeedForward(
    input_dimensions=len(problem.input_variables),
    output_dimensions=len(problem.output_variables),
    inner_size=64,
    n_layers=4,
    func=nn.Tanh,
)
```

### Template 2: Multi-scale PDE with PirateNet

```python
from pina.model import PirateNet

model = PirateNet(
    input_dimension=len(problem.input_variables),
    inner_size=64,
    output_dimension=len(problem.output_variables),
    n_layers=3,
    activation=nn.Tanh,
)
```

### Template 3: Operator learning on a grid with FNO

```python
from pina.model import FNO
from pina.model import FeedForward

model = FNO(
    lifting_net=FeedForward(input_dim, inner, 2 * n_modes),
    projecting_net=FeedForward(inner, output_dim, inner),
    n_modes=[12, 12],
    dimensions=2,
    padding=8,
)
```

### Template 4: FeedForward with Fourier Feature Embedding

```python
class FourierFeedForward(nn.Module):
    def __init__(self, input_dim, output_dim, inner_size=64, n_layers=4, sigma=2.0):
        super().__init__()
        self.embedding = FourierFeatureEmbedding(
            input_dimension=input_dim,
            output_dimension=128,
            sigma=sigma,
        )
        self.ff = FeedForward(
            input_dimensions=128,
            output_dimensions=output_dim,
            inner_size=inner_size,
            n_layers=n_layers,
        )

    def forward(self, x):
        return self.ff(self.embedding(x))

model = FourierFeedForward(
    input_dim=len(problem.input_variables),
    output_dim=len(problem.output_variables),
)
```

### Template 5: Periodic problem with embedding

```python
from pina.model.block import PeriodicBoundaryEmbedding

class PeriodicModel(nn.Module):
    def __init__(self, input_dim, output_dim, periods, inner_size=64):
        super().__init__()
        self.embedding = PeriodicBoundaryEmbedding(
            input_dimension=input_dim,
            periods=periods,
        )
        embedding_dim = input_dim * 3  # default: cos+sin per dim + 1
        self.ff = FeedForward(
            input_dimensions=embedding_dim,
            output_dimensions=output_dim,
            inner_size=inner_size,
        )

    def forward(self, x):
        return self.ff(self.embedding(x))
```

### Template 6: Custom torch.nn.Module

If no existing model fits, create a custom `nn.Module` with the same
interface pattern (labels are automatically handled by PINA solvers):

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dimensions, output_dimensions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dimensions, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, output_dimensions),
        )

    def forward(self, x):
        return self.net(x)

model = MyModel(
    input_dimensions=len(problem.input_variables),
    output_dimensions=len(problem.output_variables),
)
```

## Checklist

- [ ] Input dimensions determined (from input data in the conditions or user Q&A)
- [ ] Output dimensions determined (from output data in the conditions or user Q&A)
- [ ] Model class selected from the PINA zoo (or custom `nn.Module` if none fit)
- [ ] Model constructed with correct input and output data structure and shapes
- [ ] Model is a valid `torch.nn.Module` and compatible with the data in conditions