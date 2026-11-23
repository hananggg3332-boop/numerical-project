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
    compiled = compile(expr, "<newton_expr>", "eval")

    for name in compiled.co_names:
        if name not in ("x",) and name not in SAFE_MATH_ENV:
            raise ValueError(f"Unsupported symbol or function '{name}' in equation.")

    def _f(x_val: float) -> float:
        local_env = {"x": x_val}
        return float(eval(compiled, {"__builtins__": {}}, {**SAFE_MATH_ENV, **local_env}))

    return _f


def solve_newton(
    func: Union[str, Callable[[float], float]],
    x0: Any,
    dfunc: Optional[Union[str, Callable[[float], float]]] = None,
    tol: Any = 1e-6,
    max_iter: Any = 100,
) -> Dict[str, Any]:
    """Newton-Raphson: x_{n+1} = x_n - f(x_n) / f'(x_n).
    If `dfunc` is None, f'(x) is approximated numerically (central difference).
    """
    steps: List[Dict[str, Any]] = []

    x_val = _to_float(x0)
    if x_val is None:
        return {
            "ok": False,
            "error": "Initial guess 'x0' must be a valid number.",
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
        if isinstance(max_iter, bool):
            raise ValueError
        max_iter_int = int(max_iter)
        if max_iter_int <= 0:
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
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Error evaluating function: {exc}",
            "result": None,
            "steps": steps,
        }

    has_derivative = dfunc is not None and not (isinstance(dfunc, str) and not dfunc.strip())
    if has_derivative:
        try:
            df = _build_callable(dfunc)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Error evaluating derivative: {exc}",
                "result": None,
                "steps": steps,
            }
    else:
        def df(x_val_inner: float) -> float:
            h = 1e-6 * max(1.0, abs(x_val_inner))
            return (f(x_val_inner + h) - f(x_val_inner - h)) / (2.0 * h)

    for k in range(1, max_iter_int + 1):
        try:
            fx = f(x_val)
            dfx = df(x_val)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Error evaluating function at iteration {k}: {exc}",
                "result": None,
                "steps": steps,
            }

        if _to_float(fx) is None or _to_float(dfx) is None:
            return {
                "ok": False,
                "error": f"Function or derivative produced invalid value (NaN/Inf) at iteration {k}.",
                "result": None,
                "steps": steps,
            }

        if abs(dfx) < 1e-14:
            return {
                "ok": False,
                "error": f"Derivative is zero (or too close to zero) at iteration {k}; cannot divide by zero.",
                "result": None,
                "steps": steps,
            }

        x_new = x_val - fx / dfx

        if _to_float(x_new) is None:
            return {
                "ok": False,
                "error": f"Iteration {k} produced an invalid value (NaN/Inf). The method may be diverging.",
                "result": None,
                "steps": steps,
            }

        err_est = abs(x_new - x_val)

        steps.append(
            {
                "iteration": k,
                "summary": f"Iter {k}: x = {x_val:.6f}, f(x) = {fx:.6e}, f'(x) = {dfx:.6e}, x_new = {x_new:.6f}",
                "details": f"x_n={x_val:.6f}, f(x_n)={fx:.6e}, f'(x_n)={dfx:.6e}, x_(n+1)={x_new:.6f}",
                "x": x_val,
                "fx": fx,
                "dfx": dfx,
                "x_new": x_new,
                "error_est": err_est,
            }
        )

        if abs(fx) == 0 or err_est < tol_val:
            return {
                "ok": True,
                "error": None,
                "result": x_new,
                "steps": steps,
            }

        x_val = x_new

    return {
        "ok": False,
        "error": "Method did not converge within max_iter iterations.",
        "result": x_val,
        "steps": steps,
    }