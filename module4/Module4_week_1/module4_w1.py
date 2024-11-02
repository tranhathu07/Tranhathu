#Dataset
import numpy as np

# Load dataset
data = np.genfromtxt('advertising.csv', delimiter=',', skip_header=1)

# Get the number of samples
N = data.shape[0]

# Separate features (X) and target variable (y)
X = data[:, :3]  # Assuming the first three columns are features
y = data[:, 3]   # Assuming the fourth column is the target variable

# Normalize input data by using mean normalization
def mean_normalization(X):
    N = len(X)
    maxi = np.max(X, axis=0)  # Max along columns for multi-dimensional data
    mini = np.min(X, axis=0)  # Min along columns for multi-dimensional data
    avg = np.mean(X, axis=0)  # Mean along columns for multi-dimensional data
    X = (X - avg) / (maxi - mini)  # Normalize
    X_b = np.c_[np.ones((N, 1)), X]  # Add bias term
    return X_b, maxi, mini, avg

# Apply mean normalization
X_b, maxi, mini, avg = mean_normalization(X)

def stochastic_gradient_descent(X_b, y, n_epochs=50, learning_rate=0.00001):
    # Khởi tạo theta với giá trị ngẫu nhiên
    # thetas = np.random.randn(4, 1)  # uncomment this line for real application
    thetas = np.asarray([[1.16270837], [-0.81960489], [1.39501033], [0.29763545]])

    thetas_path = [thetas]
    losses = []
    N = len(y)

    for epoch in range(n_epochs):
        for i in range(N):
            # Chọn số ngẫu nhiên trong N
            # random_index = np.random.randint(N)  # In real application, you should use this code
            random_index = i  # This code is used for this assignment only
            
            xi = X_b[random_index: random_index + 1]
            yi = y[random_index: random_index + 1]

            # Tính toán đầu ra
            y_hat = xi @ thetas  # Sử dụng toán tử ma trận @ để nhân ma trận

            # Tính toán tổn thất li (Mean Squared Error)
            li = np.mean((y_hat- yi) ** 2)  # Tổn thất cho mẫu hiện tại

            # Tính gradient cho tổn thất
            gradient = 2 * (y_hat - yi) @ xi  # Đạo hàm của tổn thất

            # Cập nhật theta
            thetas = thetas - learning_rate * gradient.T  # Cập nhật trọng số

            # Ghi nhận tổn thất
            losses.append(li)
            thetas_path.append(thetas)

    return thetas_path, losses
agd_theta,losses = stochastic_gradient_descent(X_b,y,n_epochs = 1,learning_rate = 0.01)
# print(np.sum(losses))


def mini_batch_gradient_descent(X_b, y, n_epochs=50, minibatch_size=20, learning_rate=0.01):
    # Khởi tạo theta với giá trị ngẫu nhiên
    # thetas = np.random.randn(4, 1)  # uncomment this line for real application
    thetas = np.asarray([[1.16270837], [-0.81960489], [1.39501033], [0.29763545]])

    thetas_path = [thetas]
    losses = []
    N = len(y)
    shuffled_indices = np.asarray([
        21, 144, 17, 107, 37, 115, 167, 31, 3, 132, 179, 155, 36, 191, 182, 170,
        27, 35, 162, 25, 28, 73, 172, 152, 102, 16, 185, 11, 1, 34, 177, 29, 96,
        22, 76, 196, 6, 128, 114, 117, 111, 43, 57, 126, 165, 78, 151, 104, 110,
        53, 181, 113, 173, 75, 23, 161, 85, 94, 18, 148, 190, 169, 149, 79, 138,
        20, 108, 137, 93, 192, 198, 153, 4, 45, 164, 26, 8, 131, 77, 80, 130, 127,
        125, 61, 10, 175, 143, 87, 33, 50, 54, 97, 9, 84, 188, 139, 195, 72, 64,
        194, 44, 109, 112, 60, 86, 90, 140, 171, 59, 199, 105, 41, 147, 92, 52,
        124, 71, 197, 163, 98, 189, 103, 51, 39, 180, 74, 145, 118, 38, 47, 174,
        100, 184, 183, 160, 69, 91, 82, 42, 89, 81, 186, 136, 63, 157, 46, 67,
        129, 120, 116, 32, 19, 187, 70, 141, 146, 15, 58, 119, 12, 95, 0, 40,
        83, 24, 168, 150, 178, 49, 159, 7, 193, 48, 30, 14, 121, 5, 142, 65,
        176, 101, 55, 133, 13, 106, 66, 99, 68, 135, 158, 88, 62, 166, 156, 2,
        134, 56, 123, 122, 154
    ])
    for epoch in range(n_epochs):
        # Xáo trộn chỉ số
        shuffled_indices = np.random.permutation(N)  # uncomment this code for real application
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, N, minibatch_size):
            xi = X_b_shuffled[i:i + minibatch_size]
            yi = y_shuffled[i:i + minibatch_size]

            # Tính toán đầu ra
            y_pred = xi @ thetas  # Nhân ma trận

            # Tính toán tổn thất
            loss = np.mean((y_pred - yi) ** 2)  # Tổn thất trung bình

            # Tính đạo hàm của tổn thất
            error = yi-y_pred  # Sai số
            gradients = - (xi.T @ error) / minibatch_size  # Đạo hàm của tổn thất


            # Cập nhật tham số
            thetas = thetas - learning_rate * gradients
            thetas_path.append(thetas)

            # Ghi nhận tổn thất trung bình
            losses.append(loss)

    return thetas_path, losses

import matplotlib.pyplot as plt
agd_theta,losses = mini_batch_gradient_descent( X_b , y , n_epochs =50 ,minibatch_size = 20 , learning_rate =0.01)
x_axis = list(range(200))
plt.plot(x_axis,losses[:200],color = 'r')
plt.show()
print (round(sum(losses),2))