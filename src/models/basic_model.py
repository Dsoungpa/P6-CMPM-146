from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Define a sequential model here
model = models.Sequential()
# add more layers to the model...

# CNN: 7x7[](maybe)
# Add a conv layer
model.add(layers.Conv2D(input_shape=(20, 150, 150, 3), activation='relu', filters=2, kernel_size=7))

# Add a maxpooling layer
model.add(layers.MaxPooling2D(input_shape=(20, 150, 150, 3), pool_size=(7, 7)))


# Add a single flattened layer
model.add(layers.Flatten())

# Add a hidden densely connected layer
model.add(layers.Dense(activation='relu'))

# Add a final densely connected layer
model.add(layers.Dense(activation='sigmoid'))


# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Finally, train this compiled model by running:
# python train.py