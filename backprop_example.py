np.random.seed(0)


input_node = 1
hidden_node = 3
output_node = 1

n_train = 20
train_x = np.linspace(0, np.pi * 2, n_train)
train_y = np.sin(train_x)

n_test = 60
test_x = np.linspace(0, np.pi*2, n_test)
test_y = np.sin(test_x)

# learning rate
alpha = 0.1
max_iter = 5000

w1 = np.random.rand(hidden_node, input_node)
b1 = np.random.rand(hidden_node, 1)
w2 = np.random.rand(output_node, hidden_node)
b2 = np.random.rand(output_node, 1)

for iter in range(1, max_iter):
    # for all x in X
    for i in range(n_train):
        z1 = np.dot(w1, train_x[i].reshape(1,1)) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(w2, a1) + b2
        y_hat = z2
        y_hat_list[i] = y_hat
        e = 0.5 * (train_y[i] - y_hat) ** 2
        dy = -(train_y[i] - y_hat)
        dz2 = 1
        dw2 = a1.T
        delta_w2 = dy*dz2 + dw2
        delta_b2 = dy*dz2
        da1 = w2.T
        dz1 = d_sigmoid(z1)
        dw1 = train_x[i].T
        delta_w1 = dy*dz2*da1*dz1*dw1
        detla_b1 = dy*dz2*da1*dz1

        # backpropagation
        w2 -= alpha * delta_w2
        b2 -= alpha * delta_b2
        w1 -= alpha * delta_w1
        b1 -= alpha * delta_b1 
