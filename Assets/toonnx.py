import tf2onnx
import os

# --- Configuration ---
model_dir = "./pretrained_models"
model_name = "dtln_aec_128"
opset_version = 13 # A common and stable opset

# --- Define model paths ---
model_1_tflite_path = os.path.join(model_dir, f"{model_name}_1.tflite")
model_1_onnx_path = os.path.join(model_dir, f"{model_name}_1.onnx")

model_2_tflite_path = os.path.join(model_dir, f"{model_name}_2.tflite")
model_2_onnx_path = os.path.join(model_dir, f"{model_name}_2.onnx")


# --- Perform Conversions ---
print(f"--- Converting {os.path.basename(model_1_tflite_path)} ---")
# Use the built-in tflite converter from tf2onnx
try:
    os.system(f"python -m tf2onnx.convert --tflite \"{model_1_tflite_path}\" --output \"{model_1_onnx_path}\" --opset {opset_version}")
    print(f"Successfully converted and saved to {model_1_onnx_path}\n")
except Exception as e:
    print(f"An error occurred: {e}")


print(f"--- Converting {os.path.basename(model_2_tflite_path)} ---")
try:
    os.system(f"python -m tf2onnx.convert --tflite \"{model_2_tflite_path}\" --output \"{model_2_onnx_path}\" --opset {opset_version}")
    print(f"Successfully converted and saved to {model_2_onnx_path}\n")
except Exception as e:
    print(f"An error occurred: {e}")