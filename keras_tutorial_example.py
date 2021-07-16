from kears.models import Sequential
from keras.layers import Dense

from keras.optimizers import SGD, RMSprop, Adam
from keras import metrics
np.random.seed(0)

train_x = np.linspace(0,np.pi*2,20).reshape(20,1)
train_y = np.sin(train_x)

test_x = np.linspace(0,np.pi*2,20).reshape(60,1)
test_y = np.sin(test_x)

input_node = 1
hidden1_node = 3
output_node = 1

model = Sequential()
model.add(Dense(hidden1_node, input_dim=input_node, activation='sigmoid'))
model.add(Dense(output_node))

optimzier_option = {'sgd':SGD(lr=0.1), 'momentum':SGD(lr=alpha, momentum=0.9), 'RMSProp':RMSprop(lr=0.01), 'Adam':Adam(lr=0.1)}


result = []
train_y_hat = []

test_y_hat = []
for optimizer_name, optimizer_setting in optimizer_option.items():
    model.compile(optimizer = optimizer_setting, loss='mean_squared_error', metrics=['mse'])
    hist = model.fit(train_x, train_y, epochs= 50000, verbose=0)
    result.append(hist)
    train_result = model.predict(train_x)
    train_y_hat.append(train_result)
    test_result = mode.preidct(test_x)
    test_y_hat.append(test_result) 
