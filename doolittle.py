from typing import Any, Dict, List, Optional


def _validate_square_matrix(matrix: Any) -> Optional[str]:
    """التحقق من أن المصفوفة قائمة ثنائية الأبعاد ومربعة وتحتوي على أرقام صالحة."""
    if not isinstance(matrix, list) or len(matrix) == 0:
        return "Matrix A must be a non-empty list of lists."

    n = len(matrix)
    for row_idx, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != n:
            return f"Matrix A must be square ({n}x{n}); issue found at row {row_idx}."
        for col_idx, val in enumerate(row):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                return f"Non-numeric value at A[{row_idx}][{col_idx}]."
            if val != val or val in (float("inf"), float("-inf")):
                return f"Invalid (NaN/Inf) value at A[{row_idx}][{col_idx}]."
    return None


def doolittle_lu(A: List[List[float]]) -> Dict[str, Any]:
    """
    تحلل مصفوفة مربعة A إلى مصفوفتين L و U بطريقة Doolittle (حيث diag(L) = 1).
    ترجع قاموساً موحداً يحتوي على: ok, error, result, steps.
    """
    steps: List[Dict[str, Any]] = []

    validation_error = _validate_square_matrix(A)
    if validation_error is not None:
        return {
            "ok": False,
            "error": validation_error,
            "result": None,
            "steps": steps,
        }

    n = len(A)
    L: List[List[float]] = [
        [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)
    ]
    U: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]

    step_counter = 1

    for i in range(n):
        # 1. حساب عناصر الصف i في المصفوفة العلوية U
        for k in range(i, n):
            dot_product = sum(L[i][s] * U[s][k] for s in range(i))
            U[i][k] = float(A[i][k]) - dot_product

        steps.append({
            "iteration": step_counter,
            "summary": f"Compute row {i} of upper matrix U",
            "details": f"U[{i}][{i}..{n - 1}] = {[round(U[i][k], 6) for k in range(i, n)]}",
        })
        step_counter += 1

        # التأكد من أن العنصر القطري في U غير صفري لتجنب القسمة على صفر
        if abs(U[i][i]) < 1e-12:
            return {
                "ok": False,
                "error": (
                    f"Zero or near-zero pivot encountered at U[{i}][{i}] = {U[i][i]}. "
                    "Doolittle decomposition without pivoting cannot continue."
                ),
                "result": None,
                "steps": steps,
            }

        # 2. حساب عناصر العمود i أسفل القطر في المصفوفة السفلية L
        if i < n - 1:
            computed_l_entries = []
            for j in range(i + 1, n):
                dot_product = sum(L[j][s] * U[s][i] for s in range(i))
                L[j][i] = (float(A[j][i]) - dot_product) / U[i][i]
                computed_l_entries.append(f"L[{j}][{i}]={L[j][i]:.6f}")

            steps.append({
                "iteration": step_counter,
                "summary": f"Compute column {i} below diagonal in lower matrix L",
                "details": ", ".join(computed_l_entries),
            })
            step_counter += 1

    return {
        "ok": True,
        "error": None,
        "result": {"L": L, "U": U},
        "steps": steps,
    }