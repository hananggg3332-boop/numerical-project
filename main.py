import tkinter as tk
from tkinter import messagebox, scrolledtext
from typing import Any, Dict

from methods.gauss_seidel import solve_gauss_seidel
from methods.bisection import solve_bisection


METHODS: Dict[str, Dict[str, Any]] = {
    "Gauss-Seidel": {
        "type": "linear_system",
        "solver": solve_gauss_seidel,
    },
    "Bisection": {
        "type": "nonlinear_root",
        "solver": solve_bisection,
    },
}


def _parse_matrix(text: str):
    rows = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
        rows.append(parts)
    return rows


def _parse_vector(text: str):
    parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    return parts


def run_method(method_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    if method_name not in METHODS:
        return {
            "ok": False,
            "error": f"Unknown method '{method_name}'.",
            "result": None,
            "steps": [],
        }

    info = METHODS[method_name]
    solver = info["solver"]
    m_type = info["type"]

    if m_type == "linear_system":
        A = _parse_matrix(inputs.get("matrix_a", ""))
        b = _parse_vector(inputs.get("vector_b", ""))
        x0_raw = inputs.get("x0", "").strip()
        x0 = _parse_vector(x0_raw) if x0_raw else None
        return solver(
            A=A,
            b=b,
            x0=x0,
            tol=inputs.get("tol", 1e-6),
            max_iter=inputs.get("max_iter", 100),
        )

    if m_type == "nonlinear_root":
        return solver(
            func=inputs.get("func", ""),
            a=inputs.get("a", ""),
            b=inputs.get("b", ""),
            tol=inputs.get("tol", 1e-6),
            max_iter=inputs.get("max_iter", 100),
        )

    return {
        "ok": False,
        "error": "Unsupported method type.",
        "result": None,
        "steps": [],
    }


def launch_app() -> None:
    root = tk.Tk()
    root.title("Numerical Methods Calculator")
    root.geometry("760x620")

    tk.Label(root, text="Select Method:").pack(anchor="w", padx=12, pady=(10, 2))
    selected_method = tk.StringVar(value="Gauss-Seidel")
    tk.OptionMenu(root, selected_method, *METHODS.keys()).pack(anchor="w", padx=12)

    # --- Linear System Inputs ---
    linear_frame = tk.LabelFrame(root, text="Linear System (A*x = b)")
    linear_frame.pack(fill="x", padx=12, pady=8)

    tk.Label(linear_frame, text="Matrix A (one row per line, e.g. 4 -1 0):").pack(anchor="w", padx=8)
    txt_a = tk.Text(linear_frame, height=4, width=60)
    txt_a.pack(padx=8, pady=2)
    txt_a.insert("1.0", "4 -1 0\n-1 4 -1\n0 -1 4")

    tk.Label(linear_frame, text="Vector b (e.g. 15 10 10):").pack(anchor="w", padx=8)
    ent_b = tk.Entry(linear_frame, width=40)
    ent_b.pack(anchor="w", padx=8, pady=2)
    ent_b.insert(0, "15 10 10")

    tk.Label(linear_frame, text="Initial guess x0 (optional, e.g. 0 0 0):").pack(anchor="w", padx=8)
    ent_x0 = tk.Entry(linear_frame, width=40)
    ent_x0.pack(anchor="w", padx=8, pady=2)

    # --- Nonlinear Root Inputs ---
    root_frame = tk.LabelFrame(root, text="Nonlinear Root (f(x) = 0)")
    root_frame.pack(fill="x", padx=12, pady=8)

    tk.Label(root_frame, text="Function f(x) (e.g. x**3 - x - 2):").pack(anchor="w", padx=8)
    ent_func = tk.Entry(root_frame, width=45)
    ent_func.pack(anchor="w", padx=8, pady=2)
    ent_func.insert(0, "x**3 - x - 2")

    interval_row = tk.Frame(root_frame)
    interval_row.pack(anchor="w", padx=8, pady=2)
    tk.Label(interval_row, text="a:").pack(side="left")
    ent_a = tk.Entry(interval_row, width=12)
    ent_a.pack(side="left", padx=(2, 10))
    ent_a.insert(0, "1")

    tk.Label(interval_row, text="b:").pack(side="left")
    ent_b_root = tk.Entry(interval_row, width=12)
    ent_b_root.pack(side="left", padx=(2, 10))
    ent_b_root.insert(0, "2")

    # --- Common Parameters ---
    params_row = tk.Frame(root)
    params_row.pack(anchor="w", padx=12, pady=4)
    tk.Label(params_row, text="tol:").pack(side="left")
    ent_tol = tk.Entry(params_row, width=12)
    ent_tol.pack(side="left", padx=(2, 12))
    ent_tol.insert(0, "1e-6")

    tk.Label(params_row, text="max_iter:").pack(side="left")
    ent_max_iter = tk.Entry(params_row, width=10)
    ent_max_iter.pack(side="left", padx=(2, 12))
    ent_max_iter.insert(0, "50")

    # --- Output Area ---
    lbl_error = tk.Label(root, text="", fg="red", wraplength=720, justify="left")
    lbl_error.pack(anchor="w", padx=12)

    lbl_result = tk.Label(root, text="Result: -", fg="green", font=("Arial", 11, "bold"))
    lbl_result.pack(anchor="w", padx=12, pady=2)

    txt_steps = scrolledtext.ScrolledText(root, height=10, width=85)
    txt_steps.pack(padx=12, pady=(2, 10))

    def on_solve() -> None:
        lbl_error.config(text="")
        lbl_result.config(text="Result: -")
        txt_steps.delete("1.0", tk.END)

        m_name = selected_method.get()
        payload = {
            "matrix_a": txt_a.get("1.0", tk.END),
            "vector_b": ent_b.get(),
            "x0": ent_x0.get(),
            "func": ent_func.get(),
            "a": ent_a.get(),
            "b": ent_b_root.get(),
            "tol": ent_tol.get(),
            "max_iter": ent_max_iter.get(),
        }

        res = run_method(m_name, payload)
        if not res["ok"]:
            err_msg = res["error"] or "Unknown error."
            lbl_error.config(text=f"Error: {err_msg}")
            messagebox.showerror("Calculation Error", err_msg)
            return

        lbl_result.config(text=f"Result: {res['result']}")
        for step in res.get("steps", []):
            txt_steps.insert(tk.END, step.get("summary", str(step)) + "\n")

    tk.Button(root, text="= SOLVE", command=on_solve, padx=16, pady=4).pack(pady=4)
    root.mainloop()


if __name__ == "__main__":
    launch_app()