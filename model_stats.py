import tensorflow as tf

# Load the model
model_path = 'Pesticide_Sprayer.h5'  # Update this path if necessary
model = tf.keras.models.load_model(model_path)

# Build the model with a sample input
sample_input = tf.keras.Input(shape=(224, 224, 3))  # Adjust shape as needed
model(sample_input)

# Now you can access input properties
input_shape = model.input_shape
input_dtype = model.input.dtype

print(f"Model loaded from: {model_path}")
print(f"Input shape: {input_shape}")
print(f"Input data type: {input_dtype}")

# Print a summary of the model architecture
print("\nModel Summary:")
model.summary()

# Additional check for preprocessing requirements
print("\nNote: This model expects input values to be rescaled to the range [0, 1].")
print("Make sure to divide your input image pixel values by 255 before feeding them to the model.")
