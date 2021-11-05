from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Define a sequential model here
model = models.Sequential()
# add more layers to the model...

# CNN: 7x7[](maybe)

# Add a maxpooling layer
# Add a conv layer
input_shape = (20, 150, 150, 3)
model.add(layers.Conv2D(input_shape=input_shape[1:], activation='relu', filters=16, kernel_size=11))    # 140
# Add a maxpool
model.add(layers.MaxPooling2D(pool_size=2))     # 70
model.add(layers.MaxPooling2D(pool_size=2))     # 35
model.add(layers.MaxPooling2D(pool_size=5))     # 7
# Add a single flattened layer
model.add(layers.Flatten())

# Add a hidden densely connected layer
model.add(layers.Dense(units=16, activation='relu'))

# Add a final densely connected layer
model.add(layers.Dense(units=1, activation='sigmoid'))
model.summary()

# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Finally, train this compiled model by running:
# python train.py