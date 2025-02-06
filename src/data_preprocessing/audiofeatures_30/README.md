# 📌 Audio Feature Extraction (MP4 to WAV to CSV)

## 📂 Overview
This project extracts audio features from an MP4 file by:
1. **Converting MP4 to WAV** using FFmpeg
2. **Extracting features using Short-Time Fourier Transform (STFT)**
3. **Resizing features with Bilinear Interpolation**
4. **Saving the extracted features as a CSV file**

This ensures a structured and efficient way to process audio data for further analysis.

---

## 🚀 Installation
Before running the script, install the required dependencies:

⚠️ **FFmpeg is required for MP4 to WAV conversion.**  
### ✅ Install FFmpeg on Windows

1. Visit the [FFmpeg official download page](https://ffmpeg.org/download.html)
2. Click "Get packages & executable files" → Select Windows  
3. Download the latest `ffmpeg.zip` from **"Gyan.dev"** or **"BtbN"**
4. Extract the files and add the `bin/ffmpeg.exe` path to the system environment variable (`PATH`)
5. Verify FFmpeg installation in the terminal:

    ```bash
    ffmpeg -version
    ```
📌 If installed correctly, version information will be displayed.




