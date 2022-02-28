import tensorflow as tf

print('Using TensorFlow Version', tf.__version__)

mnist = tf.keras.datasets.mnist
print(mnist)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train)

# feature scaling
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
print("Model:", model)

print(x_train[:1])
predictions = model(x_train[:1]).numpy()
print('Predictions:', predictions)

# convert to probabilities
probabilities = tf.nn.softmax(predictions).numpy()
print("Probabilities:", probabilities)

# calculate the loss on predections
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print("Loss Function:", loss_fn)
# check by an example loss
loss = loss_fn(y_train[:1], predictions).numpy()
print("Loss:", loss)

# define the accuracy
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# train the model by adjusting the paremeters and minimizing the loss
model.fit(x_train, y_train, epochs=5)

# check the models performance on test set
model.evaluate(x_test,  y_test, verbose=2)

# get probabilities on test set
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
probability_model(x_test[:5])

