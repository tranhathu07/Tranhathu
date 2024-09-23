####(XGBoost cho bài toán Regression)######
import numpy as np

# Data
X = np.array([22, 23, 24, 25, 26, 27])  # Input data
Y = np.array([12, 14, 16, 18, 20, 22])  # Target data

# Hyperparameters
lambda_ = 0  # Regularization parameter
lr = 0.3     # Learning rate
depth = 1    # Tree depth

# Step 1: Initialize f0 as the mean of Y
f0 = np.mean(Y)
print(f"Initial prediction f0: {f0}")

# Step 2: Calculate residuals (Y - f0) and Similarity Score for root
residuals = Y - f0
sum_of_residuals = np.sum(residuals)
n = len(Y)
similarity_score_root = (sum_of_residuals ** 2) / (n + lambda_)
print(f"Similarity Score for root: {similarity_score_root}")

# Step 3: Split the data based on X thresholds and calculate Similarity Scores for each node
splits = [23.5, 25, 26.5]  # Given conditions
gains = []

for split in splits:
    left_mask = X < split
    right_mask = X >= split

    # Calculate residuals for left and right nodes
    left_residuals = residuals[left_mask]
    right_residuals = residuals[right_mask]

    # Calculate Similarity Scores for left and right nodes
    left_similarity_score = (np.sum(left_residuals) ** 2) / (len(left_residuals) + lambda_)
    right_similarity_score = (np.sum(right_residuals) ** 2) / (len(right_residuals) + lambda_)

    # Step 4: Calculate Gain
    gain = left_similarity_score + right_similarity_score - similarity_score_root
    gains.append(gain)
    print(f"Gain for split X < {split}: {gain}")

# Step 5: Select the split with the highest Gain
best_split_index = np.argmax(gains)
best_split = splits[best_split_index]
print(f"Best split is X < {best_split} with Gain = {gains[best_split_index]}")

# Step 5: Calculate Output for left and right nodes
left_mask = X < best_split
right_mask = X >= best_split

left_output = np.sum(residuals[left_mask]) / len(left_residuals)
right_output = np.sum(residuals[right_mask]) / len(right_residuals)

print(f"Left node output: {left_output}")
print(f"Right node output: {right_output}")

# Step 6: Make a prediction for x = 25
x_new = 25
if x_new < best_split:
    output = left_output
else:
    output = right_output

prediction = f0 + lr * output
print(f"Prediction for x = {x_new}: {prediction}")

########(XGBoost cho bài toán Classification)####
import numpy as np

# Data
X = np.array([22, 23, 24, 25, 26, 27])  # Input data
Y = np.array([0, 1, 0, 1, 1, 0])       # Target data (0: False, 1: True)

# Hyperparameters
lambda_ = 0  # Regularization parameter
lr = 0.3     # Learning rate
depth = 1    # Tree depth
previous_prob = 0.5  # Initial prediction probability
n = len(Y)

# Step 1: Initialize f0 with 0.5 (initial prediction)
f0 = previous_prob
print(f"Initial probability f0: {f0}")

# Step 2: Calculate residuals (Y - f0) and Similarity Score for root
residuals = Y - f0
sum_of_residuals = np.sum(residuals)
P = previous_prob
similarity_score_root = (sum_of_residuals ** 2) / (n * P * (1 - P) + lambda_)
print(f"Similarity Score for root: {similarity_score_root}")

# Step 3: Split the data based on X thresholds and calculate Similarity Scores for each node
splits = [23.5, 25, 26.5]  # Given conditions
gains = []

for split in splits:
    left_mask = X < split
    right_mask = X >= split

    # Calculate residuals for left and right nodes
    left_residuals = residuals[left_mask]
    right_residuals = residuals[right_mask]

    # Calculate Similarity Scores for left and right nodes
    left_similarity_score = (np.sum(left_residuals) ** 2) / (len(left_residuals) * P * (1 - P) + lambda_)
    right_similarity_score = (np.sum(right_residuals) ** 2) / (len(right_residuals) * P * (1 - P) + lambda_)

    # Step 4: Calculate Gain
    gain = left_similarity_score + right_similarity_score - similarity_score_root
    gains.append(gain)
    print(f"Gain for split X < {split}: {gain}")

# Step 5: Select the split with the highest Gain
best_split_index = np.argmax(gains)
best_split = splits[best_split_index]
print(f"Best split is X < {best_split} with Gain = {gains[best_split_index]}")

# Step 5: Calculate Output for left and right nodes
left_mask = X < best_split
right_mask = X >= best_split

left_output = np.sum(left_residuals) / (len(left_residuals) * P * (1 - P))
right_output = np.sum(right_residuals) / (len(right_residuals) * P * (1 - P))

print(f"Left node output: {left_output}")
print(f"Right node output: {right_output}")

# Step 6: Make a prediction for x = 25
x_new = 25
if x_new < best_split:
    output = left_output
else:
    output = right_output

# LogPrediction calculation
log_prediction = np.log(P / (1 - P)) + lr * output

# Probability calculation
prediction_probability = np.exp(log_prediction) / (1 + np.exp(log_prediction))
print(f"Predicted probability for x = {x_new}: {prediction_probability}")


####(XGBoost Regressor)#######
# (a) Import các thư viện cần thiết
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# (b) Load dữ liệu từ file CSV
dataset_path = 'Problem3.csv'  # Thay bằng đường dẫn thực tế tới tệp của bạn
data_df = pd.read_csv(dataset_path)
print(data_df.head())  # Hiển thị vài dòng đầu tiên của dữ liệu

# (c) Encode các cột dạng categorical (month, day) và boolean (rain) về dạng số
categorical_cols = data_df.select_dtypes(include=['object', 'bool']).columns.to_list()
for col_name in categorical_cols:
    n_categories = data_df[col_name].nunique()
    print(f'Number of categories in {col_name}: {n_categories}')

ordinal_encoder = OrdinalEncoder()
encoded_categorical_cols = ordinal_encoder.fit_transform(data_df[categorical_cols])

encoded_categorical_df = pd.DataFrame(encoded_categorical_cols, columns=categorical_cols)
numerical_df = data_df.drop(categorical_cols, axis=1)

# Kết hợp lại các cột số và cột đã được mã hóa
encoded_df = pd.concat([numerical_df, encoded_categorical_df], axis=1)
print(encoded_df.head())  # Hiển thị DataFrame sau khi mã hóa

# (d) Tách dữ liệu thành X và y
X = encoded_df.drop(columns=['area'])  # Biến 'area' là target/label
y = encoded_df['area']

# (e) Chia tập dữ liệu thành tập train và test (tỷ lệ 7:3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# (f) Xây dựng mô hình XGBoost Regression với các tham số đã cho
xg_reg = xgb.XGBRegressor(seed=7, learning_rate=0.01, n_estimators=102, max_depth=3)

# Huấn luyện mô hình
xg_reg.fit(X_train, y_train)

# (g) Dự đoán trên tập test và tính toán các độ đo MAE và MSE
preds = xg_reg.predict(X_test)

# Tính Mean Absolute Error (MAE) và Mean Squared Error (MSE)
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)

print('Evaluation results on test set:')
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

# Hiển thị một số kết quả dự đoán và giá trị thực tế
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
print(results_df.head())

# Vẽ biểu đồ so sánh giữa giá trị thực tế và dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
plt.xlabel('Actual Area')
plt.ylabel('Predicted Area')
plt.title('Actual vs Predicted Area (XGBoost Regression)')
plt.show()

####(XGBoost Classifier)#####
# Import các thư viện cần thiết
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# (b) Load dữ liệu từ file CSV
dataset_path = '/content/Problem4.csv'
data_df = pd.read_csv(dataset_path)

# Kiểm tra một vài dòng dữ liệu
print(data_df.head())

# (d) Tách dữ liệu thành X (input features) và y (target labels)
X, y = data_df.iloc[:, :-1], data_df.iloc[:, -1]

# (e) Chia dữ liệu thành tập train và test theo tỷ lệ 7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# (f) Xây dựng mô hình XGBoost cho Classification
xg_class = xgb.XGBClassifier(seed=7)

# Huấn luyện mô hình
xg_class.fit(X_train, y_train)

# (g) Dự đoán trên tập test
preds = xg_class.predict(X_test)

# Đánh giá mô hình sử dụng độ chính xác (Accuracy)
train_acc = accuracy_score(y_train, xg_class.predict(X_train))
test_acc = accuracy_score(y_test, preds)

# In kết quả đánh giá
print(f'Train Accuracy: {train_acc}')
print(f'Test Accuracy: {test_acc}')