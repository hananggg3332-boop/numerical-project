from typing import Any, Dict, List, Optional, Tuple


def validate_thomas_inputs(
    a: Any, b: Any, c: Any, d: Any
) -> Tuple[Optional[str], List[float], List[float], List[float], List[float]]:
    """
    Validates inputs for Thomas algorithm.
    a: lower diagonal (length n-1)
    b: main diagonal (length n)
    c: upper diagonal (length n-1)
    d: right-hand side vector (length n)
    """
    if (
        not isinstance(a, list)
        or not isinstance(b, list)
        or not isinstance(c, list)
        or not isinstance(d, list)
    ):
        return "All inputs (a, b, c, d) must be lists.", [], [], [], []

    n = len(b)
    if n == 0:
        return "Main diagonal 'b' cannot be empty.", [], [], [], []

    if len(d) != n:
        return f"Vector 'd' length ({len(d)}) must match matrix size ({n}).", [], [], [], []

    if len(a) != n - 1 or len(c) != n - 1:
        return (
            f"Diagonals 'a' and 'c' must have length n-1 ({n-1}), but got len(a)={len(a)} and len(c)={len(c)}.",
            [],
            [],
            [],
            [],
        )

    try:
        a_vec = [float(x) for x in a]
        b_vec = [float(x) for x in b]
        c_vec = [float(x) for x in c]
        d_vec = [float(x) for x in d]
    except (ValueError, TypeError):
        return "All elements in a, b, c, and d must be numeric.", [], [], [], []

    return None, a_vec, b_vec, c_vec, d_vec


def thomas_algorithm(
    a: List[float], b: List[float], c: List[float], d: List[float]
) -> Dict[str, Any]:
    """
    Solves a tridiagonal system Ax = d using the Thomas algorithm.
    Returns steps, success status, and final solution vector x.
    """
    err, a_vec, b_vec, c_vec, d_vec = validate_thomas_inputs(a, b, c, d)
    if err:
        return {"success": False, "error": err, "solution": None, "steps": []}

    n = len(b_vec)
    c_star = [0.0] * (n - 1)
    d_star = [0.0] * n
    x = [0.0] * n
    steps = []

    if b_vec[0] == 0:
        return {"success": False, "error": "Division by zero at index 0 (b[0] = 0).", "solution": None, "steps": []}

    c_star[0] = c_vec[0] / b_vec[0]
    d_star[0] = d_vec[0] / b_vec[0]

    steps.append({
        "step": 0,
        "type": "forward",
        "c_star": c_star[0],
        "d_star": d_star[0]
    })

    for i in range(1, n):
        denom = b_vec[i] - a_vec[i - 1] * c_star[i - 1]
        if denom == 0:
            return {
                "success": False,
                "error": f"Division by zero during forward elimination at index {i}.",
                "solution": None,
                "steps": steps,
            }

        if i < n - 1:
            c_star[i] = c_vec[i] / denom

        d_star[i] = (d_vec[i] - a_vec[i - 1] * d_star[i - 1]) / denom

        steps.append({
            "step": i,
            "type": "forward",
            "c_star": c_star[i] if i < n - 1 else None,
            "d_star": d_star[i]
        })

    x[n - 1] = d_star[n - 1]
    steps.append({"step": n - 1, "type": "backward", "x_val": x[n - 1]})

    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]
        steps.append({"step": i, "type": "backward", "x_val": x[i]})

    return {
        "success": True,
        "error": None,
        "solution": x,
        "steps": steps
    }


if __name__ == "__main__":
    sample_a = [-1, -1, -1]
    sample_b = [4, 4, 4, 4]
    sample_c = [-1, -1, -1]
    sample_d = [5, 5, 5, 5]

    result = thomas_algorithm(sample_a, sample_b, sample_c, sample_d)
    print("Success:", result["success"])
    print("Solution x:", result["solution"])
    print("Total steps recorded:", len(result["steps"]))