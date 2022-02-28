from keras.models import Model
from keras.layers import Input
from keras.layers import Dense

# define input layer using the feature structure
input_layer = Input(shape=(3,)) # this has the size of the feature vector

# connect hidden layer with input layer
hidden_layer = Dense(3)(input_layer)

# connect output layer with previous hidden layer
output_layer = Dense(1)(hidden_layer)

# create the model and pass in input and output layer
model = Model(inputs=input_layer, outputs=output_layer)

print(model.summary())


