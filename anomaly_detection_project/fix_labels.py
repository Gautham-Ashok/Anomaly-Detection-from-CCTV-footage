import numpy as np

label_path = "E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\balanced_labels.npy"

# ✅ Load labels
y = np.load(label_path)

# ✅ Convert to float32
y = y.astype(np.float32)

# ✅ Save the fixed labels
np.save(label_path, y)

# ✅ Check new properties
print("✅ Updated Labels Shape:", y.shape)  # Should be (6000,)
print("✅ Updated Labels Type:", y.dtype)  # Should be float32
