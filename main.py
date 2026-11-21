from jacobi import jacobi_method

# الحالة 1: عنصر صفري على القطر الرئيسي
A_zero_diag = [
    [0.0, -1.0, 2.0],
    [-1.0, 11.0, -1.0],
    [2.0, -1.0, 10.0],
]
b_3 = [6.0, 25.0, -11.0]
x0_3 = [0.0, 0.0, 0.0]

res1 = jacobi_method(A_zero_diag, b_3, x0_3, tol=1e-6, max_iterations=50)
print("Test 1 (Zero diagonal) -> ok:", res1["ok"], "| error:", res1["error"])

# الحالة 2: عدم التقارب خلال عدد تكرارات صغير جداً (max_iterations=2)
A_conv = [
    [10.0, -1.0, 2.0, 0.0],
    [-1.0, 11.0, -1.0, 3.0],
    [2.0, -1.0, 10.0, -1.0],
    [0.0, 3.0, -1.0, 8.0],
]
b_4 = [6.0, 25.0, -11.0, 15.0]
x0_4 = [0.0, 0.0, 0.0, 0.0]

res2 = jacobi_method(A_conv, b_4, x0_4, tol=1e-6, max_iterations=2)
print("Test 2 (Max iterations reached) -> ok:", res2["ok"], "| error:", res2["error"])

# الحالة 3: طول المتجه b لا يطابق حجم المصفوفة A
res3 = jacobi_method(A_conv, [6.0, 25.0], x0_4, tol=1e-6, max_iterations=50)
print("Test 3 (Size mismatch) -> ok:", res3["ok"], "| error:", res3["error"])