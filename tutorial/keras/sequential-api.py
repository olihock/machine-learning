from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=3, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
