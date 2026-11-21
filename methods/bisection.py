import math
from typing import Any, Callable, Dict, List, Optional, Union


SAFE_MATH_ENV: Dict[str, Any] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
    "pi": math.pi,
    "e": math.e,
}


def _to_float(val: Any) -> Optional[float]:
    if isinstance(val, bool):
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if num != num or num in (float("inf"), float("-inf")):
        return None
    return num


def _build_callable(func: Union[str, Callable[[float], float]]) -> Callable[[float], float]:
    if callable(func):
        return func

    if not isinstance(func, str) or not func.strip():
        raise ValueError("Equation/function cannot be empty.")

    expr = func.strip().replace("^", "**")
    compiled = compile(expr, "<bisection_expr>", "eval")

    for name in compiled.co_names:
        if name not in ("x",) and name not in SAFE_MATH_ENV:
            raise ValueError(f"Unsupported symbol or function '{name}' in equation.")

    def _f(x_val: float) -> float:
        local_env = {"x": x_val}
        return float(eval(compiled, {"__builtins__": {}}, {**SAFE_MATH_ENV, **local_env}))

    return _f


def solve_bisection(
    func: Union[str, Callable[[float], float]],
    a: Any,
    b: Any,
    tol: Any = 1e-6,
    max_iter: Any = 100,
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []

    a_val = _to_float(a)
    b_val = _to_float(b)
    if a_val is None or b_val is None:
        return {
            "ok": False,
            "error": "Interval bounds 'a' and 'b' must be valid numbers.",
            "result": None,
            "steps": steps,
        }

    if a_val >= b_val:
        return {
            "ok": False,
            "error": "Lower bound 'a' must be strictly less than upper bound 'b'.",
            "result": None,
            "steps": steps,
        }

    tol_val = _to_float(tol)
    if tol_val is None or tol_val <= 0:
        return {
            "ok": False,
            "error": "Tolerance (tol) must be a positive number.",
            "result": None,
            "steps": steps,
        }

    try:
        max_iter_int = int(max_iter)
        if max_iter_int <= 0 or isinstance(max_iter, bool):
            raise ValueError
    except (TypeError, ValueError):
        return {
            "ok": False,
            "error": "max_iter must be a positive integer.",
            "result": None,
            "steps": steps,
        }

    try:
        f = _build_callable(func)
        fa = f(a_val)
        fb = f(b_val)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Error evaluating function: {exc}",
            "result": None,
            "steps": steps,
        }

    if _to_float(fa) is None or _to_float(fb) is None:
        return {
            "ok": False,
            "error": "Function returned an invalid value (NaN/Inf) at interval endpoints.",
            "result": None,
            "steps": steps,
        }

    if abs(fa) == 0:
        return {
            "ok": True,
            "error": None,
            "result": a_val,
            "steps": [
                {
                    "iteration": 0,
                    "summary": f"Root found at a = {a_val:.6f} where f(a) = 0.",
                    "details": f"a={a_val:.6f}, b={b_val:.6f}, f(a)={fa:.6e}",
                    "c": a_val,
                    "fc": fa,
                }
            ],
        }

    if abs(fb) == 0:
        return {
            "ok": True,
            "error": None,
            "result": b_val,
            "steps": [
                {
                    "iteration": 0,
                    "summary": f"Root found at b = {b_val:.6f} where f(b) = 0.",
                    "details": f"a={a_val:.6f}, b={b_val:.6f}, f(b)={fb:.6e}",
                    "c": b_val,
                    "fc": fb,
                }
            ],
        }

    if fa * fb > 0:
        return {
            "ok": False,
            "error": "Invalid interval: f(a) and f(b) must have opposite signs (f(a) * f(b) < 0).",
            "result": None,
            "steps": steps,
        }

    curr_a, curr_b = a_val, b_val
    prev_c: Optional[float] = None

    for k in range(1, max_iter_int + 1):
        c = (curr_a + curr_b) / 2.0
        try:
            fc = f(c)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Error evaluating function at iteration {k}: {exc}",
                "result": None,
                "steps": steps,
            }

        if _to_float(fc) is None:
            return {
                "ok": False,
                "error": f"Function produced invalid value (NaN/Inf) at iteration {k}.",
                "result": None,
                "steps": steps,
            }

        err_est = abs(c - prev_c) if prev_c is not None else abs(curr_b - curr_a) / 2.0

        steps.append(
            {
                "iteration": k,
                "summary": f"Iter {k}: c = {c:.6f}, f(c) = {fc:.6e}, [{curr_a:.4f}, {curr_b:.4f}]",
                "details": f"a={curr_a:.6f}, b={curr_b:.6f}, c={c:.6f}, f(c)={fc:.6e}",
                "a": curr_a,
                "b": curr_b,
                "c": c,
                "fc": fc,
                "error_est": err_est,
            }
        )

        if abs(fc) == 0 or (prev_c is not None and err_est < tol_val):
            return {
                "ok": True,
                "error": None,
                "result": c,
                "steps": steps,
            }

        if fa * fc < 0:
            curr_b = c
            fb = fc
        else:
            curr_a = c
            fa = fc

        prev_c = c

    return {
        "ok": True,
        "error": None,
        "result": prev_c,
        "steps": steps,
    }