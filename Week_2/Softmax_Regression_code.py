import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset_path = './Week_2/creditcard.csv'
df = pd.read_csv(dataset_path)

dataset_arr = df.to_numpy()
X,y = dataset_arr[:,:-1].astype(np.float64),dataset_arr[:,-1].astype(np.uint8)

intercept = np.ones((X.shape[0],1))

X_b = np.concatenate((intercept,X),axis = 1)

n_classes = np.unique(y,axis = 0).shape[0]
n_samples=y.shape[0]
y_encoded = np.array([np.zeros(n_classes) for _ in range(n_samples)])
y_encoded[np.arange(n_samples),y] = 1

val_size = 0.2
test_size = 0.125
random_state = 2
is_shuffle = True

X_train,X_val,y_train, y_val = train_test_split(X_b,y_encoded,test_size= val_size,random_state=random_state,shuffle = is_shuffle)

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size= test_size,random_state= random_state,shuffle= is_shuffle)

normalizer = StandardScaler()
X_train[:,1:] = normalizer.fit_transform(X_train[:,1:])
X_test[:,1:] = normalizer.transform(X_test[:,1:])
X_val[:,1:] = normalizer.transform(X_val[:,1:])

#hàm softmax
def softmax(z):
    exp_z = np.exp(z)
    return exp_z/exp_z.sum(axis = 1)[:,None]

def predict(X,theta):
    x = np.dot(X,theta)
    y_hat = softmax(x)
    return y_hat
#Xây dựng hàm tính loss với công thức Cross-entropy
def compute_loss(y_hat,y):
    n = y.size
    return (-1/n) * np.sum(y*np.log(y_hat))

def compute_gradient(X,y,y_hat):
    n = y.size
    return np.dot(X.T,(y_hat - y))/n
def update_theta(theta,gradient,lr):
    theta = theta - lr*gradient
    return theta
def compute_accuracy(X,y,theta):
    y_hat = predict(X,theta)
    acc = (np.argmax(y_hat,axis = 1) == np.argmax(y,axis = 1)).mean()
    return acc

lr = 0.1
epochs =200
batch_size = X_train.shape[0]
n_features = X_train.shape[1]
np.random.seed(random_state)
theta = np.random.uniform(size = (n_features,n_classes))

train_losses = []
train_accs = []
val_losses = []
val_accs =[]
for epoch in range(epochs):
    train_batch_losses = []
    train_batch_accs = []
    val_batch_losses = []
    val_batch_accs = []
    for i in range(0,X_train.shape[0],batch_size):
        xi = X_train[i:i+batch_size]
        yi = y_train[i:i+batch_size]

        y_hat = predict(xi,theta)
        loss = compute_loss(y_hat,yi)
        train_batch_losses.append(loss)

        gradient = compute_gradient(xi,yi,y_hat)
        theta = update_theta(theta,gradient,lr)
        acc = compute_accuracy(X_train,y_train,theta)
        train_batch_accs.append(acc)

        y_val_hat = predict(X_val,theta)
        loss_val = compute_loss(y_val_hat,y_val)
        val_batch_losses.append(loss_val)
        acc_val = compute_accuracy(X_val,y_val,theta)
        val_batch_accs.append(acc_val)
    train_batch_loss = np.mean(train_batch_losses)
    train_batch_acc = np.mean(train_batch_accs)
    val_batch_acc = np.mean(val_batch_accs)
    val_batch_loss = np.mean(val_batch_losses)

    train_losses.append(train_batch_loss)
    train_accs.append(train_batch_acc)
    val_losses.append(val_batch_loss)
    val_accs.append(val_batch_acc)
    print (f'\ nEPOCH { epoch + 1}:\ tTraining loss : { train_batch_loss :.3f}\ tValidation loss : { val_batch_loss :.3f}')

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0, 0].plot(train_losses, color='green')
ax[0, 0].set(xlabel='Epoch', ylabel='Loss')
ax[0, 0].set_title('Training Loss')

ax[0, 1].plot(val_losses, color='orange')
ax[0, 1].set(xlabel='Epoch', ylabel='Loss')
ax[0, 1].set_title('Validation Loss')

ax[1, 0].plot(train_accs, color='green')
ax[1, 0].set(xlabel='Epoch', ylabel='Accuracy')
ax[1, 0].set_title('Training Accuracy')

ax[1, 1].plot(val_accs, color='orange')
ax[1, 1].set(xlabel='Epoch', ylabel='Accuracy')
ax[1, 1].set_title('Validation Accuracy')

plt.show()

# Val set
val_set_acc = compute_accuracy(X_val, y_val, theta)
print('Evaluation on validation set:')
print(f'Accuracy: {val_set_acc}')

