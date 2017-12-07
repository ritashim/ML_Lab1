from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from numpy import zeros, insert, ones, r_

def compute_loss(x, y, theta):
    n = y.shape[0]
    l = x.dot(theta)-y
    r = (l.T.dot(l))/2/n
    return r

# Load data set
x, y = load_svmlight_file("housing_scale")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train_n = y_train.shape[0]


# Parameters, all-zero preset
learning_rate = 0.01
theta = zeros((13,1))
iter_num = 1000

# Do the calculation
loss_history = zeros((iter_num,1))

for iter in range(iter_num):
    theta = theta - (learning_rate/y_train_n) * (x_train.T.dot(x_train.dot(theta)-y_train))
    loss_history[iter] = compute_loss(x_test, y_test, theta)
    print ("Iter %d, Loss: %.2f"%(iter,loss_history[iter]))
