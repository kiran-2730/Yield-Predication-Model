import joblib
import os

# Original model file
model_path = "finalNew_trained_crop_yield_model.pkl"

# Load old pickle model
model = joblib.load(model_path)

# Save compressed model
joblib.dump(
    model,
    "compressed_model.joblib",
    compress=3
)

# Show sizes
old_size = os.path.getsize(model_path) / (1024 * 1024)
new_size = os.path.getsize("compressed_model.joblib") / (1024 * 1024)

print(f"Original Size: {old_size:.2f} MB")
print(f"Compressed Size: {new_size:.2f} MB")