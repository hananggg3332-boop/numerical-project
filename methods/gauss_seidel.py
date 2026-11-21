from typing import Any, Dict, List, Optional


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


def solve_gauss_seidel(
    A: List[List[Any]],
    b: List[Any],
    x0: Optional[List[Any]] = None,
    tol: Any = 1e-6,
    max_iter: Any = 100,
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []

    if not isinstance(A, list) or len(A) == 0:
        return {
            "ok": False,
            "error": "Matrix A must be a non-empty list of lists.",
            "result": None,
            "steps": steps,
        }

    n = len(A)
    parsed_A: List[List[float]] = []

    for i, row in enumerate(A):
        if not isinstance(row, list) or len(row) != n:
            return {
                "ok": False,
                "error": f"Matrix A must be square ({n}x{n}); check row {i + 1}.",
                "result": None,
                "steps": steps,
            }
        parsed_row: List[float] = []
        for j, val in enumerate(row):
            num = _to_float(val)
            if num is None:
                return {
                    "ok": False,
                    "error": f"Invalid numeric value in A at row {i + 1}, column {j + 1}.",
                    "result": None,
                    "steps": steps,
                }
            parsed_row.append(num)
        parsed_A.append(parsed_row)

    if not isinstance(b, list) or len(b) != n:
        return {
            "ok": False,
            "error": f"Vector b must be a list of {n} numbers.",
            "result": None,
            "steps": steps,
        }

    parsed_b: List[float] = []
    for idx, val in enumerate(b):
        num = _to_float(val)
        if num is None:
            return {
                "ok": False,
                "error": f"Invalid numeric value in vector b at position {idx + 1}.",
                "result": None,
                "steps": steps,
            }
        parsed_b.append(num)

    if x0 is None:
        x = [0.0] * n
    else:
        if not isinstance(x0, list) or len(x0) != n:
            return {
                "ok": False,
                "error": f"Initial guess x0 must be a list of {n} numbers.",
                "result": None,
                "steps": steps,
            }
        x = []
        for idx, val in enumerate(x0):
            num = _to_float(val)
            if num is None:
                return {
                    "ok": False,
                    "error": f"Invalid numeric value in x0 at position {idx + 1}.",
                    "result": None,
                    "steps": steps,
                }
            x.append(num)

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

    for i in range(n):
        if abs(parsed_A[i][i]) == 0:
            return {
                "ok": False,
                "error": f"Zero diagonal element at row {i + 1}, column {i + 1}; division by zero.",
                "result": None,
                "steps": steps,
            }

    for k in range(1, max_iter_int + 1):
        x_old = x.copy()
        eq_details: List[str] = []

        for i in range(n):
            sum_before = sum(parsed_A[i][j] * x[j] for j in range(i))
            sum_after = sum(parsed_A[i][j] * x_old[j] for j in range(i + 1, n))
            new_xi = (parsed_b[i] - sum_before - sum_after) / parsed_A[i][i]

            if new_xi != new_xi or new_xi in (float("inf"), float("-inf")):
                return {
                    "ok": False,
                    "error": f"Numerical overflow/divergence detected at iteration {k}.",
                    "result": None,
                    "steps": steps,
                }

            x[i] = new_xi
            eq_details.append(f"x{i + 1} = {new_xi:.6f}")

        max_diff = max(abs(x[i] - x_old[i]) for i in range(n))
        steps.append(
            {
                "iteration": k,
                "summary": f"Iter {k}: " + ", ".join(eq_details) + f" (diff={max_diff:.6e})",
                "details": " | ".join(eq_details),
                "values": x.copy(),
                "error_est": max_diff,
            }
        )

        if max_diff < tol_val:
            return {
                "ok": True,
                "error": None,
                "result": x,
                "steps": steps,
            }

    return {
        "ok": False,
        "error": f"Gauss-Seidel did not converge within {max_iter_int} iterations.",
        "result": x,
        "steps": steps,
    }