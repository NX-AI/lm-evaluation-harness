import ast
import numpy as np
from .utils_execute import check_correctness

# Allow these literal-ish constructs for CRUXEval-O generations:
# - constants: numbers, strings, bools, None
# - containers: list/tuple/set/dict literals (including nesting)
# - unary +/- on constants (e.g. -1)
# - basic binary ops on literal-ish operands (e.g. "a" + "b", 1 + 2)
_ALLOWED_EXPR_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)

# Explicitly disallow these even if other nodes might allow them indirectly
_DISALLOWED_NODES = (
    ast.Call,          # f(x), len(...), etc.
    ast.Attribute,     # obj.attr
    ast.Subscript,     # a[0]
    ast.Lambda,        # lambda ...
    ast.IfExp,         # x if cond else y
    ast.Compare,       # ==, <, etc.
    ast.BoolOp,        # and/or
    ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp,  # comprehensions
    ast.Await, ast.Yield, ast.YieldFrom,
)

_ALLOWED_NAMES = {"True", "False", "None"}

def _require_literalish_expr(expr: str) -> None:
    """
    Enforce that `expr` is a "literal-ish" Python expression:
      - No function calls, no attribute access, no variable names.
      - Allows literals + literal containers + simple arithmetic/string concat.
    This matches the prompt constraint "literal (no unsimplified expressions, no function calls)"
    while remaining robust to nested literals.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid Python expression: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ValueError(f"Disallowed name: {node.id}")
        if isinstance(node, _DISALLOWED_NODES):
            raise ValueError(f"Disallowed construct: {type(node).__name__}")
        if not isinstance(node, _ALLOWED_EXPR_NODES):
            # Catch-all for anything we didn't explicitly allow
            raise ValueError(f"Unsupported construct: {type(node).__name__}")

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k.
    n: total samples
    c: correct samples
    """
    if n <= 0:
        return 0.0
    k = min(k, n)
    if c <= 0:
        return 0.0
    if c >= n:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def evaluate_score(args):
    gs, (code, inp, out), mode = args

    execution_results = []
    for g in gs:
        # 1) Basic empty check
        if not g or not str(g).strip():
            execution_results.append(False)
            continue

        g = str(g).strip()

        # 2) Construct the validation code based on task type
        if mode == "input":
            # CRUXEval-I: model generates input expression(s) to satisfy f(x) == out
            check_program = f"{code}\nassert f({g}) == ({out})"

        elif mode == "output":
            # CRUXEval-O: model generates an output expression; keep it literal-ish
            try:
                _require_literalish_expr(g)
            except Exception:
                execution_results.append(False)
                continue

            # Validate: generated expression evaluates to same value as ground truth expression
            check_program = f"{code}\nassert ({g}) == ({out})"

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 3) Execute
        execution_results.append(check_correctness(check_program, timeout=3))

    return execution_results


def evaluate_doc_generations(gs, code, inp, out, mode):
    """
    gs: list of generation strings for a single CRUXEval sample
    Returns pass@1 up to pass@n_repeats in powers of 2.
    """
    execution_results = evaluate_score((gs, (code, inp, out), mode))

    c = execution_results.count(True)
    n = len(execution_results)

    powers_of_2 = [1]
    while powers_of_2[-1] * 2 <= n:
        powers_of_2.append(powers_of_2[-1] * 2)

    return {f"pass_at_{k}": pass_at_k(n, c, k) for k in powers_of_2}

