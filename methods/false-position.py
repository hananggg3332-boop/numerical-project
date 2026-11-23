import math
from typing import Callable, Dict, List, Optional


def false_position(
    func: Callable[[float], float],
    xl: float,
    xu: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Dict[str, object]:
    """
    Solves f(x) = 0 using the False Position (Regula Falsi) method.
    """
    if xl >= xu:
        return {
            "success": False,
            "error": "xl must be less than xu.",
            "root": None,
            "steps": [],
        }

    fl = func(xl)
    fu = func(xu)

    if fl * fu > 0:
        return {
            "success": False,
            "error": "f(xl) and f(xu) must have opposite signs.",
            "root": None,
            "steps": [],
        }

    steps: List[Dict[str, Optional[float]]] = []
    xr = xl
    prev_xr: Optional[float] = None

    for i in range(1, max_iter + 1):
        if (fu - fl) == 0:
            return {
                "success": False,
                "error": "Division by zero (f(xu) - f(xl) == 0).",
                "root": None,
                "steps": steps,
            }

        xr = xu - (fu * (xl - xu)) / (fu - fl)
        fr = func(xr)

        approx_error: Optional[float] = None
        if prev_xr is not None and xr != 0:
            approx_error = abs((xr - prev_xr) / xr) * 100.0

        steps.append(
            {
                "iter": i,
                "xl": xl,
                "xu": xu,
                "xr": xr,
                "f(xl)": fl,
                "f(xu)": fu,
                "f(xr)": fr,
                "error_%": approx_error,
            }
        )

        if fr == 0 or (approx_error is not None and approx_error < tol):
            return {
                "success": True,
                "error": None,
                "root": xr,
                "iterations": i,
                "steps": steps,
            }

        if fl * fr < 0:
            xu = xr
            fu = fr
        else:
            xl = xr
            fl = fr

        prev_xr = xr

    return {
        "success": True,
        "error": "Reached max_iter before reaching tolerance.",
        "root": xr,
        "iterations": max_iter,
        "steps": steps,
    }


if __name__ == "__main__":
    # Example: f(x) = x^3 - x - 2
    def f(x: float) -> float:
        return x**3 - x - 2

    res = false_position(f, xl=1.0, xu=2.0, tol=1e-6, max_iter=100)
    print("Success:", res["success"])
    print("Root:", res["root"])
    print("Iterations:", res["iterations"])