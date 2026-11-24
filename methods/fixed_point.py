import math


def fixed_point(g, x0, tolerance=1e-6, max_iterations=100):
    steps = []

    if not callable(g):
        return {
            "ok": False,
            "error": "g must be a valid function.",
            "result": None,
            "steps": steps,
        }

    try:
        x0 = float(x0)
        tolerance = float(tolerance)
    except (ValueError, TypeError):
        return {
            "ok": False,
            "error": "Initial value and tolerance must be numeric.",
            "result": None,
            "steps": steps,
        }

    if not math.isfinite(x0):
        return {
            "ok": False,
            "error": "Initial value must be finite.",
            "result": None,
            "steps": steps,
        }

    if not math.isfinite(tolerance) or tolerance <= 0:
        return {
            "ok": False,
            "error": "Tolerance must be a positive finite number.",
            "result": None,
            "steps": steps,
        }

    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        return {
            "ok": False,
            "error": "Maximum iterations must be a positive integer.",
            "result": None,
            "steps": steps,
        }

    x_old = x0

    for i in range(1, max_iterations + 1):
        try:
            x_new = float(g(x_old))

            if not math.isfinite(x_new):
                return {
                    "ok": False,
                    "error": "Function produced a non-finite value.",
                    "result": None,
                    "steps": steps,
                }
        except Exception as e:
            return {
                "ok": False,
                "error": f"Function evaluation error: {e}",
                "result": None,
                "steps": steps,
            }

        error = abs(x_new - x_old)

        steps.append(
            {
                "iteration": i,
                "summary": f"Iteration {i}: Calculate new approximation.",
                "details": (
                    f"x{i} = g(x{i-1}) = {x_new:.10f}, "
                    f"Error = |{x_new:.10f} - {x_old:.10f}| "
                    f"= {error:.10f}"
                ),
            }
        )

        if error < tolerance:
            return {
                "ok": True,
                "error": None,
                "result": x_new,
                "steps": steps,
            }

        x_old = x_new

    return {
        "ok": False,
        "error": "Method did not converge within maximum iterations.",
        "result": x_old,
        "steps": steps,
    }