from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Define your dropout model here
model = models.Sequential()

# CNN: 7x7[]

# Add a conv layer
input_shape = (20, 150, 150, 3)
model.add(layers.Conv2D(input_shape=input_shape[1:], activation='relu', filters=128, kernel_size=3))    # 148
# Add a maxpool
model.add(layers.MaxPooling2D(pool_size=2))     # 74
# Conv2D
model.add(layers.Conv2D(activation='relu', filters=64, kernel_size=5))    # 70
# Maxpools
model.add(layers.MaxPooling2D(pool_size=2))     # 35
model.add(layers.MaxPooling2D(pool_size=5))     # 7
# Add a single flattened layer
model.add(layers.Flatten())

# Add a dropout layer
model.add(layers.Dropout(rate=0.2))

# Add a hidden densely connected layer
model.add(layers.Dense(units=64, activation='relu'))

# Add a final densely connected layer
model.add(layers.Dense(units=1, activation='sigmoid'))
model.summary()
# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Train this compiled model by modifying basic_train 
# to import this model, then run:
#   python train.py