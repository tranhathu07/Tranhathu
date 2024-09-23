from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# Load Iris dataset
iris_X, iris_y = datasets.load_iris(return_X_y=True)

# Split train : test = 8:2
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y,
    test_size=0.2,
    random_state=42
)

# Define model
dt_classifier = DecisionTreeClassifier()

# Train
dt_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the result
print(f"Accuracy: {accuracy}")

###MSE###
# Load dataset
machine_cpu = fetch_openml(name='machine_cpu')
machine_data = machine_cpu.data
machine_labels = machine_cpu.target

# Split train : test = 8:2
X_train, X_test, y_train, y_test = train_test_split(
    machine_data, machine_labels,
    test_size=0.2,
    random_state=42
)

# Define model
tree_reg = DecisionTreeRegressor()

# Train
tree_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = tree_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Print the result
print(f"Mean Squared Error: {mse}")
