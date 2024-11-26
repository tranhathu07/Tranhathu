import numpy as np
lr = 0.4
W = [-5,-2]
def notusenumpy(W,epochs,lr):
    for _ in range(epochs):
        dw = [0] *len(W)
        for i in range(len(W)):
            dw[i] = 2*W[i]
        for u in range(len(W)):
            W[u] = W[u]-lr*dw[u]
    return W
print(f'Câu a:{notusenumpy(W,2,lr)}')

def dfw(W):
    dw = [0] *len(W)
    for i in range(len(W)):
        dw[i] = 2*W[i]
    return dw
def sgd(W,dw,lr):
    for i in range(len(W)):
        W[i] = W[i]-lr*dw[i]
    return W
def train_pl(sgd,lr,epochs):
    W = np.array([-5,-2],dtype=np.float32)
    result = [W.copy()]
    for _ in range(epochs):
        dw = dfw(W)
        W = sgd(W,dw,lr)
    result.append(W.copy())
    return result
results = train_pl(sgd,lr,30)
print(f"Câu b: w1 = {results[-1][0]:.4f}, w2 = {results[-1][1]:.4f}")


# Hàm train
def train_pl(sgd, lr, epochs):
    W = [-5.0, -2.0]  # Khởi tạo W là danh sách Python
    results = [W.copy()]  # Lưu kết quả qua từng epoch
    for _ in range(epochs):
        dw = dfw(W)
        W = sgd(W, dw, lr)
        results.append(W.copy())  # Lưu kết quả sau mỗi epoch
    return results


# ###Bài 2 Gradient Descent + Momentum ###
b=0.5
epochs=2
def notusenp(W,epochs,lr,b):
    v = [0] *(len(W)+1)
    for _ in range(epochs):
        dw = [0] *len(W)
        for i in range(len(W)):
            dw[i] = 2*W[i]
        for t in range(len(W)):
            v[t+1] = b*(v[t]) + (1-b)*dw[t]
            W[t] = W[t] - lr*(v[t+1])
    return W
print(notusenp(W,epochs,lr,b))     

def sgd_momentum(W,dw,lr):
    v = [0] *(len(W)+1)
    for t in range(len(W)):
        v[t+1] = b*(v[t]) + (1-b)*dw[t]
        W[t] = W[t] - lr*(v[t+1])
    return v,W
W=[-5,-2]
V=[0,0]
epochs = 2
def train_gdm(W,lr,epochs):
    W = np.array([-5,-2],dtype=np.float32)
    result = [W]
    for _ in range(epochs):
        dw = dfw(W)
        [w1,w2] = sgd_momentum(W,dw,lr)
        result.append([w1,w2])
    return result

print(train_gdm(W,lr,epochs))


#####Bài 3#####
# def dfw(W):
#     dw = [0] *len(W)
#     for i in range(len(W)):
#         dw[i] = 2*W[i]
#     return dw

# def rmsprop(W,lr,epochs,gamma,epsilon=1e-8):
#     s=[0] *(len(W))
#     for epoch in range(epochs):
#         dw = dfw(W)
#         for i in range(len(W)):
#             s[i] = gamma *s[i] +(1-gamma)*(dw[i])**2
#             W[i]= W[i] - lr * dw[i] / (np.sqrt(s[i]) + epsilon)
#     return W,s

# lr = 0.3
# gamma = 0.9
# epochs = 2
# W = [-5.0, -2.0]  # Khởi tạo W ban đầu


# W_final, s_final = rmsprop(W,lr,epochs,gamma,epsilon=1e-6)


# print(f"Điểm tối ưu cau a: W = {W_final}")

# lr = 0.1 
# gamma = 0.9
# epochs_b = 30
# W = [-5.0, -2.0]  # Khởi tạo W ban đầu


# W_final_b, s_final_b = rmsprop(W,lr,epochs_b,gamma,epsilon=1e-6)


# print(f"Điểm tối ưu cau b: W = {W_final_b}")


# ##Bài 4########
# def dfw(W):
#     dw = [0] *len(W)
#     for i in range(len(W)):
#         dw[i] = 2*W[i]
#     return dw
# def adam(W, V, S, dw, lr=0.2, beta1=0.9, beta2=0.999, epsilon=1e-6, t=1):
#     # Cập nhật V và S
#     V = np.array(V)
#     S = np.array(S)
#     dw = np.array(dw)
#     V_new = beta1 * V + (1 - beta1) * dw
#     S_new = beta2 * S + (1 - beta2) * (dw ** 2)

#     # Bias correction
#     V_corr = V_new / (1 - beta1**t)
#     S_corr = S_new / (1 - beta2**t)

#     # Cập nhật weights
#     W_new = W - lr * V_corr / (np.sqrt(S_corr) + epsilon)

#     return W_new, V_new, S_new

# def train_p1(optimize, eta, epochs):
#     # Khởi tạo
#     W = np.array([-5, -2], dtype=np.float64)  # w1, w2 ban đầu
#     V = np.zeros_like(W)  # v1, v2
#     S = np.zeros_like(W)  # s1, s2
#     results = [W.tolist()]  # Lưu lại từng bước

#     for t in range(1, epochs + 1):
#         dw = dfw(W)  # Tính gradient
#         W, V, S = optimize(W, V, S, dw, eta, t=t)  # Gọi hàm Adam
#         results.append(W.tolist())  # Lưu kết quả

#     return results

# results = train_p1(adam, eta=0.01, epochs=30)

# # In kết quả cuối cùng
# print(f"Trạng thái cuối cùng: w1 = {results[-1][0]:.4f}, w2 = {results[-1][1]:.4f}")