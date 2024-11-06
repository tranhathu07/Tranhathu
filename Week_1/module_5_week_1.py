import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset_path = 'titanic_modified_dataset.csv'
df = pd.read_csv(dataset_path, index_col = 'PassengerId')

#Chia biến X và y
dataset_arr = df.to_numpy().astype(np.float64)
X,y = dataset_arr[:,:-1],dataset_arr[:,-1]

#Thêm Bias vào X
intercept = np.ones((X.shape[0],1))

X_b = np.concatenate((intercept,X),axis = 1)

#Chia tập train,val,test ( theo tỷ lệ 7:2:1)

val_size = 0.2
test_size = 0.125
random_state = 2
is_shuffle = True

X_train,X_val,y_train,y_val = train_test_split(X_b,y,test_size = val_size, random_state = random_state, shuffle = is_shuffle)
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size= test_size,shuffle= is_shuffle)
#Chuẩn hóa dữ liệu

normalizer = StandardScaler()
X_train[:,1:] = normalizer.fit_transform(X_train[:,1:])
X_val[:,1:] = normalizer.transform(X_val[:,1:])
X_test[:,1:] = normalizer.transform(X_test[:,1:])

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
#Cài đặt hàm quan trọng 

#Hàm sigmoid
def sigmoid(z):
    sigmoid =  1/(1 + np.exp(-z))
    return sigmoid
#Hàm dự đoán
def compute_predict(X,theta):
    dot_product = np.dot(X,theta)
    y_hat = sigmoid(dot_product)
    return y_hat
#Hàm tính loss
def compute_loss(y_hat,y):
    y_hat = np.clip(y_hat,1e-7,1-1e-7)
    return  (-y * np.log(y_hat) - (1-y)*np.log(1-y_hat)).mean()

#Hàm Gradient
def compute_gradient(X,y_hat,y):
    gradient = np.dot(X.T,(y_hat-y))/y.size
    return gradient

#Hàm cập nhật trọng số
def update_theta(theta,lr,gradient):
    theta = theta - lr*gradient
    return theta

#Hàm accurancy
def compute_accurancy(X,y,theta):
    y_hat = compute_predict(X,theta).round()
    acc = (y_hat == y).mean()
    return acc


lr = 0.01
epochs = 100
batch_size = 16
np.random.seed(random_state)
theta = np.random.uniform(size = X_train.shape[1])

train_accs = []
train_losses = []
val_accs = []
val_losses = []
for epoch in range(epochs):
    train_batch_accs = []
    train_batch_losses = []
    val_batch_accs =[]
    val_batch_losses = []
    for i in range(0,X_train.shape[0],batch_size):
        xi = X_train[i:i+batch_size]
        yi = y_train[i:i+batch_size]

        y_hat = compute_predict(xi,theta)
        loss = compute_loss(y_hat,yi)
        train_batch_losses.append(loss)

        gradient = compute_gradient(xi,y_hat,yi)
        theta = update_theta(theta,lr,gradient)
        acc = compute_accurancy(X_train,y_train,theta)
        train_batch_accs.append(acc)

        y_val_hat = compute_predict(X_val,theta)
        loss_val = compute_loss(y_val_hat,y_val)
        val_batch_losses.append(loss_val)
        val_acc = compute_accurancy(X_val,y_val,theta)
        val_batch_accs.append(val_acc)
    
    train_batch_loss = np.mean(train_batch_losses)
    val_batch_loss = np.mean(val_batch_losses)
    train_batch_acc = np.mean(train_batch_accs)
    val_batch_acc = np.mean(val_batch_accs)

    train_losses.append(train_batch_loss)
    train_accs.append(train_batch_acc)
    val_losses.append(val_batch_loss)
    val_accs.append(val_batch_acc)

    print(f'\nEPOCH {epoch + 1}:\tTraining loss: {train_losses[-1]:.3f}\tValidation loss: {val_losses[-1]:.3f}')
fig,ax = plt.subplots(2,2,figsize = (12,10))
ax[0,0].plot(train_losses)
ax[0,0].set(xlabel = 'Epoch',ylabel = 'Loss')
ax[0,0].set_title('Training Loss')

ax[0,1].plot(val_losses,'orange')
ax[0,1].set(xlabel = 'Epoch',ylabel = 'Loss')
ax[0,1].set_title('Validation Loss')

ax[1,0].plot(train_accs)
ax[1,0].set(xlabel = 'Epoch', ylabel = 'Accuracy')
ax[1,0].set_title('Training Accuracy')

ax[1,1].plot(val_accs,'orange')
ax[1,1].set(xlabel = 'Epoch', ylabel = 'Accuracy')
ax[1,1].set_title('Validation Accuracy')
plt.show()

val_set_acc = compute_accurancy(X_val,y_val,theta)
test_set_acc = compute_accurancy(X_test,y_test,theta)
print(f'Evaluation on validation and test set :')
print(f'Accuracy : { val_set_acc }')
print(f'Accuracy : { test_set_acc }')


### Twitter Sentiment Analysis
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import TweetTokenizer
from collections import defaultdict

dataset_path = 'sentiment_analysis.csv'
df = pd.read_csv(dataset_path,index_col = 'id')

def text_nomalize(text):
    # Retweet old acronym "RT" removal
    text = re.sub(r'^RT[\s]+','',text)

    # Hyperlinks removal
    text = re.sub(r'https?:\/\/.*','',text)

    #Hashtag removal
    text = re.sub(r'#','',text)

    #Punctuation removal
    text = re.sub(r'[^\w\s]','',text)

    #Tokenization
    tokenizer = TweetTokenizer(preserve_case = False,
    strip_handles = True,
    reduce_len = True)
    text_tokens = tokenizer.tokenize(text)
    return text_tokens
# Trong đó:
# – Dòng 1: Khai báo hàm text_normalize() nhận đầu vào là một string (text).
# – Dòng 2, 3: Loại bỏ các từ "RT"trong text (đây là một cụm từ viết tắt cũ cho "Retweet").
# – Dòng 5, 6: Loại bỏ các đường dẫn trong text.
# – Dòng 8, 9: Loại bỏ các hashtag.
# – Dòng 11, 12: Loại bỏ các dấu câu.
# – Dòng 14, 15, 16, 17, 18, 19: Khai báo tokenizer.
# – Dòng 20: Tokenize text (kết quả trả về là danh sách các token).
# – Dòng 22: Trả về danh sách các token.

#Xây dựng bộ lưu giữu tần suất xuất hiện các từ
def get_freqs(df):
    freq = defaultdict(lambda: 0)
    for idx,row in df.iterrows():
        tweet = row['tweet']
        label = row['label']
        tokens = text_nomalize(tweet)

        for token in tokens:
            pair = (label,tweet)
            freq[pair] += 1
    return freq

def get_feature(text,freq):
    tokens = text_nomalize(text)
    X= np.zeros(3)
    X[0] = 1
    for token in tokens:
        X[1] += freqs[(token,0)]
        X[2] += freqs[(token,1)]
    return X

X= []
y = []
freqs = get_freqs(df)
for idx,row in df.iterrows():
    tweet = row['tweet']
    label = row['label']

    X_i = get_feature(tweet,freqs)
    X.append(X_i)
    y.append(label)
X = np.array(X)
y = np.array(y)

val_size = 0.2
test_size = 0.125
random_state = 2
is_shuffle = True
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = val_size,random_state = random_state,shuffle = is_shuffle)
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,random_state = random_state, shuffle = is_shuffle)

normalizer = StandardScaler()
X_train[:,1:] = normalizer.fit_transform(X_train[:,1:])
X_val[:,1:] = normalizer.transform(X_val[:,1:])
X_test[:,1:] = normalizer.transform(X_test[:,1:])

def sigmoid(z):
    return 1/(1+np.exp(-z))
def compute_loss(y_hat,y):
    y_hat = np.clip(y_hat,1e-7,1-1e-7)
    return (-y* np.log(y_hat) - (1-y)*np.log(1-y_hat)).mean()
def predict(X,theta):
    dot_product = np.dot(X,theta)
    y_hat = sigmoid(dot_product)
    return y_hat
def compute_gradient(X,y,y_hat):
    return np.dot(X.T,(y_hat -y))/y.size
def update_theta(theta,gradient,lr):
    theta = theta - lr* gradient
    return theta
def compute_accurancy(X,y,theta):
    y_hat = predict(X,theta).round()
    acc = (y_hat == y).mean()
    return acc

lr = 0.01
epochs = 200
batch_size = 128

np.random.seed(random_state)
theta = np.random.uniform(size = X_train.shape[1])

train_accs = []
train_losses = []
val_accs = []
val_losses = []
for epoch in range(epochs):
    train_batch_accs = []
    train_batch_losses = []
    val_batch_accs =[]
    val_batch_losses = []
    for i in range(0,X_train.shape[0],batch_size):
        xi = X_train[i:i + batch_size]
        yi = y_train[i:i+batch_size]

        y_hat = predict(xi,theta)
        loss = compute_loss(y_hat,yi)
        train_batch_losses.append(loss)

        gradient = compute_gradient(xi,yi,y_hat)
        theta = update_theta(theta,gradient,lr)
        acc = compute_accurancy(X_train,y_train,theta)
        train_batch_accs.append(acc)

        y_val_hat = predict(X_val,theta)
        loss_val = compute_loss(y_val_hat,y_val)
        val_batch_losses.append(loss_val)
        val_acc = compute_accurancy(X_val,y_val,theta)
        val_batch_accs.append(val_acc)

    train_batch_loss = np.mean(train_batch_losses)
    val_batch_loss = np.mean(val_batch_losses)
    train_batch_acc = np.mean(train_batch_accs)
    val_batch_acc = np.mean(val_batch_accs)

    train_losses.append(train_batch_loss)
    train_accs.append(train_batch_acc)
    val_losses.append(val_batch_loss)
    val_accs.append(val_batch_acc)

print(f'\nEPOCH {epoch + 1}:\tTraining loss: {train_losses[-1]:.3f}\tValidation loss: {val_losses[-1]:.3f}')
fig,ax = plt.subplots(2,2,figsize = (12,10))
ax[0,0].plot(train_losses)
ax[0,0].set(xlabel = 'Epoch',ylabel = 'Loss')
ax[0,0].set_title('Training Loss')


ax[0,1].plot(val_losses,'orange')
ax[0,1].set(xlabel = 'Epoch',ylabel = 'Loss')
ax[0,1].set_title('Validation Loss')


ax[1,0].plot(train_accs)
ax[1,0].set(xlabel = 'Epoch', ylabel = 'Accuracy')
ax[1,0].set_title('Training Accuracy')


ax[1,1].plot(val_accs,'orange')
ax[1,1].set(xlabel = 'Epoch', ylabel = 'Accuracy')
ax[1,1].set_title('Validation Accuracy')
plt.show()


val_set_acc = compute_accurancy(X_val,y_val,theta)
test_set_acc = compute_accurancy(X_test,y_test,theta)
print(f'Evaluation on validation and test set :')
print(f'Accuracy : { val_set_acc }')
print(f'Accuracy : { test_set_acc }')
