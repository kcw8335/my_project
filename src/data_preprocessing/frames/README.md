# 📌 Video Frame Extraction

## 📂 Overview
This project extracts frames from a **video dataset**, specifically from **Training Data 1/6**, as a sample.

The purpose of this extraction is to demonstrate the process of converting video files into individual frames.

---

## 📥 Download the Dataset
You can download the video dataset from the official **ChaLearn LAP** page:  
👉 [ChaLearn LAP Dataset - Video Files](https://chalearnlap.cvc.uab.cat/dataset/24/data/41/files/)

Ensure that you have the correct dataset files before proceeding with the extraction.

---
## 📥 Extracted Data
✅ **Dataset:** Training Data **1/6** (Sample)  
✅ **Output Format:** `.jpg.npy` images  
✅ **Frame Extraction Method:** `torchvision.io.read_video()`  

---

# 📌 Execution files

### **1️⃣ Single-Processing Method** (`make_video_frames_single_processing.ipynb`)
✅ **Description**:  
- Uses a **single CPU core** to process video frames sequentially.  
- Suitable for small datasets but **slower for large-scale video processing**.  
- Implemented in a **Jupyter Notebook** for easy demonstration and experimentation.

### **2️⃣ Multi-Processing Method (Python Script)** (`make_video_frames_multi_processing.py`)
✅ **Description**:

- Uses **multiple CPU cores** to process multiple videos in parallel.
- Significantly faster than the single-processing approach, especially for large datasets.
- Recommended for real-world applications where speed is critical.