# Global Rules for All Skills

These rules apply to **every skill** in this repository.

1. **No source code modification** — Never modify, create, or delete any source
   code files (`.py`, `.cpp`, `.js`, `.ts`, `.rs`, etc.) or any files under
   `pina/`, `tests/`, `docs/`, or similar source directories.
2. **Read-only guidance** — Skills exist solely to provide guidance, templates,
   and interactive Q&A. Output explanations, code snippets for the user to
   copy, and checklists — never write files to the user's project.
3. **Scope** — If the user asks to edit or write source files, politely decline
   and explain that this skill only provides guidance.

4. **Avoid redundant operator calls in equations** — Differential operators
   (`grad`, `div`, `laplacian`, etc.) are **not cached**: every call triggers a
   new backward pass through the autograd graph.  Always compute an operator
   once for all relevant components and extract what you need, rather than
   calling it repeatedly for individual components.

   **Bad** — 6 separate backward passes:
   ```python
   Ex_t = grad(output_, input_, components=["Ex"], d=["t"])
   Ey_t = grad(output_, input_, components=["Ey"], d=["t"])
   Ez_t = grad(output_, input_, components=["Ez"], d=["t"])
   Hx_t = grad(output_, input_, components=["Hx"], d=["t"])
   Hy_t = grad(output_, input_, components=["Hy"], d=["t"])
   Hz_t = grad(output_, input_, components=["Hz"], d=["t"])
   ```

   **Good** — 1 backward pass, 6 extractions:
   ```python
   out_t = grad(output_, input_, d=["t"])
   Ex_t = out_t.extract("dExdt")
   Ey_t = out_t.extract("dEydt")
   Ez_t = out_t.extract("dEzdt")
   Hx_t = out_t.extract("dHxdt")
   Hy_t = out_t.extract("dHydt")
   Hz_t = out_t.extract("dHzdt")
   ```

   This also applies to spatial gradients, divergences, and any other
   operator that can be computed for all components at once.
