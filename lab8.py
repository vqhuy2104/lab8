# ===============================================================
# LAB 8: TỐI ƯU HÓA & QUY HOẠCH TUYẾN TÍNH  (gộp Bài 1 -> Bài 5)
# ===============================================================
import math


# ===============================================================
# BÀI 1: Giải LP 2D bằng phương pháp duyệt đỉnh hình học
# ===============================================================
def solve_lp_geometric(vertices, c1, c2):
    """Định lý cơ bản của LP: nếu có nghiệm tối ưu thì có ít nhất một nghiệm tại đỉnh
    của miền đa diện lồi => chỉ cần duyệt các đỉnh."""
    best_val = -float('inf')
    best_vertex = None

    for x1, x2 in vertices:
        current_val = c1 * x1 + c2 * x2
        print(f"Đỉnh ({x1}, {x2}) -> Giá trị f = {current_val}")
        if current_val > best_val:
            best_val = current_val
            best_vertex = (x1, x2)

    return best_vertex, best_val


def bai1():
    print("=" * 60)
    print("BÀI 1: LP 2D - duyệt đỉnh hình học")
    print("=" * 60)
    # Ràng buộc: 2x1 + x2 <= 100 ; x1 + x2 <= 80 ; x1, x2 >= 0
    vertices = [(0, 0), (0, 80), (20, 60), (50, 0)]
    c1, c2 = 40, 30
    optimal_point, max_profit = solve_lp_geometric(vertices, c1, c2)
    print(f"\nNghiệm tối ưu: {optimal_point}")      # (20, 60)
    print(f"Lợi nhuận tối đa: {max_profit} USD")    # 2600


# ===============================================================
# BÀI 2: Chuyển LP về dạng chính tắc (thêm biến bù)
# ===============================================================
def convert_to_standard_form(c, A_ub, b_ub):
    m = len(A_ub)   # số ràng buộc
    n = len(c)      # số biến gốc

    # 1. max c^T x  ->  min (-c)^T x ; biến bù có hệ số chi phí 0
    c_std = [-val for val in c] + [0.0] * m

    # 2. A_eq = [A_ub | I_m]
    A_eq = []
    for i in range(m):
        slack_row = [1.0 if j == i else 0.0 for j in range(m)]
        A_eq.append(list(A_ub[i]) + slack_row)

    # 3. Vế phải giữ nguyên
    b_eq = list(b_ub)
    return c_std, A_eq, b_eq


def bai2():
    print("\n" + "=" * 60)
    print("BÀI 2: Dạng chính tắc (Standard Form)")
    print("=" * 60)
    c = [40, 30]
    A_ub = [[2, 1], [1, 1]]
    b_ub = [100, 80]
    c_std, A_eq, b_eq = convert_to_standard_form(c, A_ub, b_ub)
    print("Hàm mục tiêu c_std:", c_std)   # [-40, -30, 0.0, 0.0]
    print("Ma trận A_eq:")
    for row in A_eq:
        print(row)                         # [2,1,1.0,0.0] ; [1,1,0.0,1.0]
    print("Vế phải b_eq:", b_eq)


# ===============================================================
# BÀI 3: Bài toán đối ngẫu (Dual) & định lý đối ngẫu mạnh
# ===============================================================
def generate_dual_problem(c_primal, A_primal, b_primal):
    """Primal: max c^T x, A x <= b, x >= 0
       Dual  : min b^T y, A^T y >= c, y >= 0"""
    m, n = len(A_primal), len(A_primal[0])
    c_dual = list(b_primal)
    A_dual = [[A_primal[i][j] for i in range(m)] for j in range(n)]  # A^T
    b_dual = list(c_primal)
    return c_dual, A_dual, b_dual


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def bai3():
    print("\n" + "=" * 60)
    print("BÀI 3: Bài toán đối ngẫu (Dual)")
    print("=" * 60)
    c = [40, 30]
    A = [[2, 1], [1, 1]]
    b = [100, 80]

    c_dual, A_dual, b_dual = generate_dual_problem(c, A, b)
    print("Dual: min", c_dual, "· y")
    print("      A_dual (= A^T) =", A_dual)
    print("      A_dual · y >=", b_dual, ", y >= 0")

    x_star = [20, 60]
    y_star = [10, 20]

    # Kiểm tra tính khả thi
    primal_ok = all(dot(A[i], x_star) <= b[i] for i in range(len(b))) and all(v >= 0 for v in x_star)
    dual_ok = all(dot(A_dual[j], y_star) >= b_dual[j] for j in range(len(b_dual))) and all(v >= 0 for v in y_star)
    print("x* khả thi (primal):", primal_ok, "| y* khả thi (dual):", dual_ok)

    primal_val = dot(c, x_star)       # c^T x*
    dual_val = dot(c_dual, y_star)    # b^T y*
    print(f"c^T x* = {primal_val}")
    print(f"b^T y* = {dual_val}")
    print("Đối ngẫu mạnh c^T x* == b^T y* == 2600:", primal_val == dual_val == 2600)
    # Ý nghĩa: y1 = 10 USD/giờ GPU, y2 = 20 USD/đơn vị VRAM là "giá bóng" (shadow price)
    # của từng tài nguyên: tăng 1 đơn vị tài nguyên thì lợi nhuận tối đa tăng đúng bấy nhiêu.


# ===============================================================
# BÀI 4: Gradient Descent 1D
# ===============================================================
def gradient_descent_1d(initial_w, learning_rate, epochs):
    """L(w) = w^2 - 4w + 4 = (w-2)^2 ; L'(w) = 2w - 4."""
    w = initial_w
    loss_history = [w ** 2 - 4 * w + 4]          # loss ban đầu
    for _ in range(epochs):
        w = w - learning_rate * (2 * w - 4)
        loss_history.append(w ** 2 - 4 * w + 4)
    return w, loss_history


def bai4():
    print("\n" + "=" * 60)
    print("BÀI 4: Gradient Descent 1D")
    print("=" * 60)
    w_star, history = gradient_descent_1d(10.0, 0.1, 20)
    for i in (0, 1, 2, 5, 10, 20):
        print(f"Epoch {i:2d}: Loss = {history[i]:.6f}")
    print(f"w* sau 20 epoch = {w_star:.4f}  (mục tiêu 2.0)")
    print(f"Loss cuối = {history[-1]:.6f}")
    # Mỗi bước: (w - 2) bị nhân với (1 - 2*lr) = 0.8  =>  w_k - 2 = 8 * 0.8^k.
    # Sau 20 bước còn 8*0.8^20 ≈ 0.09 nên w ≈ 2.09: đang tiến sát 2.0 và sẽ hội tụ
    # hẳn nếu tăng epochs (vd 100 epoch -> w ≈ 2.0000).
    w100, _ = gradient_descent_1d(10.0, 0.1, 100)
    print(f"Với 100 epoch: w* = {w100:.6f}")


# ===============================================================
# BÀI 5: Ràng buộc L1 (Lasso) vs L2 (Ridge)
# ===============================================================
def loss_fn(w1, w2, center=(3.0, 3.0)):
    return (w1 - center[0]) ** 2 + (w2 - center[1]) ** 2


def solve_constrained_l1(center=(3.0, 3.0), radius=2.0, steps=400000):
    """Duyệt các điểm trên biên hình thoi |w1| + |w2| = radius, tìm điểm có Loss nhỏ nhất.
    (Nghiệm tối ưu luôn nằm trên biên vì tâm sai số nằm ngoài miền.)"""
    best_pt, best_loss = None, float('inf')
    for k in range(steps + 1):
        w1 = -radius + 2 * radius * k / steps
        r = radius - abs(w1)
        for w2 in (r, -r):
            l = loss_fn(w1, w2, center)
            if l < best_loss:
                best_loss, best_pt = l, (w1, w2)
    return best_pt, best_loss


def solve_constrained_l2(center=(3.0, 3.0), radius=2.0, steps=400000):
    """Duyệt các điểm trên đường tròn w1^2 + w2^2 = radius^2 theo góc theta."""
    best_pt, best_loss = None, float('inf')
    for k in range(steps):
        th = 2 * math.pi * k / steps
        w1, w2 = radius * math.cos(th), radius * math.sin(th)
        l = loss_fn(w1, w2, center)
        if l < best_loss:
            best_loss, best_pt = l, (w1, w2)
    return best_pt, best_loss


def bai5():
    print("\n" + "=" * 60)
    print("BÀI 5: L1 (Lasso) vs L2 (Ridge)")
    print("=" * 60)

    # ---- L1 với tâm (3,3) đúng như đề ----
    pt, l = solve_constrained_l1((3, 3))
    print(f"[L1] tâm (3,3): nghiệm ≈ ({pt[0]:.3f}, {pt[1]:.3f}), Loss = {l:.4f}")
    print(f"     So sánh hai đỉnh trên trục: Loss(2,0) = {loss_fn(2, 0)}, Loss(0,2) = {loss_fn(0, 2)}")
    # Với tâm (3,3) nằm đúng trên đường chéo w1 = w2 nên điểm gần nhất trên cạnh
    # w1 + w2 = 2 là chân đường vuông góc (1,1), Loss = 8 < 10. Vậy trường hợp ĐỀU
    # này nghiệm KHÔNG rơi vào đỉnh trục; (2,0)/(0,2) chỉ là lệch hơn (Loss = 10).

    # ---- L1 khi tâm lệch khỏi đường chéo: xuất hiện tính thưa ----
    pt2, l2 = solve_constrained_l1((3, 1))
    print(f"[L1] tâm (3,1): nghiệm ≈ ({pt2[0]:.3f}, {pt2[1]:.3f}), Loss = {l2:.4f}  -> w2 = 0 (thưa)")

    # ---- L2 ----
    pt3, l3 = solve_constrained_l2((3, 3))
    print(f"[L2] tâm (3,3): nghiệm ≈ ({pt3[0]:.3f}, {pt3[1]:.3f}), Loss = {l3:.4f}")
    print(f"     sqrt(2) = {math.sqrt(2):.4f}  => nghiệm (sqrt2, sqrt2) ≈ (1.414, 1.414)")
    pt4, l4 = solve_constrained_l2((3, 1))
    print(f"[L2] tâm (3,1): nghiệm ≈ ({pt4[0]:.3f}, {pt4[1]:.3f})  -> cả hai trọng số đều khác 0")


# ---------------------------------------------------------------
# GIẢI THÍCH
# ---------------------------------------------------------------
# * L2 (tròn): cực tiểu của Loss trên hình tròn tại điểm mà gradient Loss cùng phương với
#   pháp tuyến của đường tròn, tức w = radius * center/||center||. Với tâm (3,3):
#   w = 2*(1,1)/sqrt(2) = (sqrt2, sqrt2) ≈ (1.414, 1.414). Mặt tròn "trơn" (không có góc)
#   nên điểm tiếp xúc hầu như không rơi đúng trục => các trọng số bị co đều theo tỉ lệ,
#   không bị triệt tiêu về 0 (weight decay: w <- (1 - lr*lambda) w).
# * L1 (thoi): hình thoi có ĐỈNH nằm trên các trục. Đường đồng mức của Loss (các vòng tròn)
#   có xu hướng chạm vào ngoài cùng ở đỉnh nhọn khi tâm lệch khỏi đường chéo, làm một trọng
#   số bằng đúng 0 => Sparsity / chọn đặc trưng (Feature Selection). Về giải tích, L1 có
#   đạo hàm không liên tục tại 0 (tạo "lực kéo" hằng số về 0), còn L2 có lực tỉ lệ với w
#   nên yếu dần khi w nhỏ.
#
# ---------------------------------------------------------------
# TỔNG KẾT MÔN HỌC (xuyên suốt 8 bài)
# ---------------------------------------------------------------
# - Đại số tuyến tính: vector, ma trận, trị riêng/vector riêng, phân rã (SVD/PCA) là ngôn ngữ
#   để biểu diễn dữ liệu và tham số mô hình (mỗi tầng mạng nơ-ron là một phép nhân ma trận).
# - Đồ thị: BFS/DFS, luồng cực đại - lát cắt cực tiểu, ghép cặp mô hình hóa quan hệ giữa các
#   thực thể (mạng xã hội, đồ thị tri thức, Graph Neural Network, lập lịch).
# - Xác suất: chuỗi Markov, trạng thái dừng, PageRank (vector riêng của P^T ứng với lambda=1)
#   nối đại số tuyến tính với đồ thị và mô hình hóa sự bất định (HMM, MCMC, random walk).
# - Tối ưu hóa: LP, đối ngẫu (nền tảng SVM), Gradient Descent và Regularization L1/L2 là cách
#   "huấn luyện" mô hình: định nghĩa hàm mất mát rồi cực tiểu hóa có ràng buộc.
# => Một mô hình AI hiện đại = dữ liệu biểu diễn bằng ma trận + cấu trúc đồ thị/xác suất +
#    hàm mất mát được tối ưu bằng gradient, có điều chuẩn để tổng quát hóa tốt.


# ===============================================================
# CHẠY TOÀN BỘ
# ===============================================================
if __name__ == "__main__":
    bai1()
    bai2()
    bai3()
    bai4()
    bai5()