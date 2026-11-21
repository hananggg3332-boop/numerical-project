from typing import Any, Dict, List, Optional


def _validate_numeric_vector(vec: Any, name: str, expected_len: Optional[int] = None) -> Optional[str]:
    """التحقق من أن المتجه قائمة أرقام صالحة وطوله مطابق للمطلوب."""
    if not isinstance(vec, list) or len(vec) == 0:
        return f"Vector '{name}' must be a non-empty list of numbers."

    if expected_len is not None and len(vec) != expected_len:
        return f"Vector '{name}' length ({len(vec)}) must match matrix size ({expected_len})."

    for idx, val in enumerate(vec):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return f"Non-numeric value in '{name}' at index {idx}."
        if val != val or val in (float("inf"), float("-inf")):
            return f"Invalid (NaN/Inf) value in '{name}' at index {idx}."

    return None


def _validate_jacobi_inputs(
    A: Any,
    b: Any,
    x0: Any,
    tol: Any,
    max_iterations: Any,
) -> Optional[str]:
    """التحقق من جميع مدخلات طريقة Jacobi والشروط الحسابية الأساسية."""
    if not isinstance(A, list) or len(A) == 0:
        return "Matrix A must be a non-empty list of lists."

    n = len(A)
    for row_idx, row in enumerate(A):
        if not isinstance(row, list) or len(row) != n:
            return f"Matrix A must be square ({n}x{n}); issue found at row {row_idx}."
        for col_idx, val in enumerate(row):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                return f"Non-numeric value at A[{row_idx}][{col_idx}]."
            if val != val or val in (float("inf"), float("-inf")):
                return f"Invalid (NaN/Inf) value at A[{row_idx}][{col_idx}]."

    # التحقق من عدم وجود أصفار على القطر الرئيسي لتجنب القسمة على صفر
    for i in range(n):
        if abs(float(A[i][i])) < 1e-12:
            return f"Zero or near-zero diagonal element at A[{i}][{i}] = {A[i][i]}; cannot divide by zero in Jacobi method."

    err_b = _validate_numeric_vector(b, "b", expected_len=n)
    if err_b is not None:
        return err_b

    err_x0 = _validate_numeric_vector(x0, "x0", expected_len=n)
    if err_x0 is not None:
        return err_x0

    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol <= 0:
        return "Tolerance 'tol' must be a positive number."

    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        return "'max_iterations' must be a positive integer."

    return None


def jacobi_method(
    A: List[List[float]],
    b: List[float],
    x0: List[float],
    tol: float = 1e-6,
    max_iterations: int = 100,
) -> Dict[str, Any]:
    """
    تحل نظام المعادلات الخطية Ax = b بطريقة Jacobi التكرارية.
    ترجع قاموساً موحداً يحتوي على: ok, error, result, steps.
    """
    steps: List[Dict[str, Any]] = []

    validation_error = _validate_jacobi_inputs(A, b, x0, tol, max_iterations)
    if validation_error is not None:
        return {
            "ok": False,
            "error": validation_error,
            "result": None,
            "steps": steps,
        }

    n = len(A)
    x_old: List[float] = [float(val) for val in x0]
    b_vec: List[float] = [float(val) for val in b]
    A_mat: List[List[float]] = [[float(val) for val in row] for row in A]

    for k in range(1, max_iterations + 1):
        x_new: List[float] = [0.0 for _ in range(n)]

        # حساب قيمة كل متغير جديدة بالاعتماد على القيم القديمة x_old فقط
        for i in range(n):
            off_diag_sum = sum(
                A_mat[i][j] * x_old[j] for j in range(n) if j != i
            )
            x_new[i] = (b_vec[i] - off_diag_sum) / A_mat[i][i]

        # حساب أكبر فرق بين التكرار الحالي والسابق (L_infinity norm)
        max_diff = max(abs(x_new[i] - x_old[i]) for i in range(n))

        steps.append({
            "iteration": k,
            "summary": f"Iteration {k}: max diff = {max_diff:.6e}",
            "details": f"x = {[round(val, 6) for val in x_new]}",
        })

        # التحقق من شرط التقارب
        if max_diff < tol:
            return {
                "ok": True,
                "error": None,
                "result": x_new,
                "steps": steps,
            }

        x_old = x_new

    return {
        "ok": False,
        "error": (
            f"Jacobi method did not converge within {max_iterations} iterations "
            f"(last max difference = {max_diff:.6e}, tolerance = {tol})."
        ),
        "result": None,
        "steps": steps,
    }