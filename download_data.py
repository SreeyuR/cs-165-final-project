import os
import gdown

# Google Drive file ID
file_id = "109FopvRlE7ZVWQq50Qr_i4JHLWTt4m5G"

# Direct download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Output file name
output = "Full_HDL_dataset_unnormalized_no_nan_column_names_w_shot_and_time.pickle"

# Download file if it doesn't already exist
if not os.path.exists(output):
    print(f"Downloading {output}...")
    gdown.download(url, output, quiet=False)
else:
    print("File already exists, skipping download.")
