import os
import torch
import torchvision.io as io
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# 🔹 Function to save a frame (used in multiprocessing)
def save_frame(args):
    frame, save_filepath = args
    np.save(save_filepath, frame)


# 🔹 Main function for processing videos (to be used with multiprocessing)
def process_videos(root_path):
    folders = os.listdir(root_path)

    # Get the number of CPU cores available (to prevent system overload)
    num_workers = max(1, cpu_count() // 2)  # Use half of the available CPU cores

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for folder in folders:
        video_filenames = os.listdir(os.path.join(root_path, folder))

        for video_filename in video_filenames:
            video_path = os.path.join(root_path, folder, video_filename)

            # 🔹 Load video using GPU if available
            video, _, info = io.read_video(video_path, pts_unit="sec")
            video = video.to(device)  # Move video to GPU

            # Create output directory for frames
            output_folder = os.path.join("./frames", video_filename)
            os.makedirs(output_folder, exist_ok=True)

            num_frames = video.shape[0]
            print(f"🔹 Processing {video_filename} ({num_frames} frames)...")

            # 🔹 Convert frames to NumPy and prepare for multiprocessing (move from GPU → CPU)
            save_tasks = []
            for i in range(num_frames):
                frame = (
                    video[i].permute(2, 0, 1).cpu().numpy()
                )  # Convert GPU → CPU → NumPy
                save_filename = f"{i:06d}.jpg.npy"
                save_filepath = os.path.join(output_folder, save_filename)
                save_tasks.append((frame, save_filepath))

            # 🔹 Use multiprocessing to save frames in parallel
            with Pool(num_workers) as pool:
                list(
                    tqdm(
                        pool.imap_unordered(save_frame, save_tasks),
                        total=num_frames,
                        desc="Saving Frames",
                        leave=True,
                    )
                )

            print(f"✅ {num_frames} frames saved in {output_folder}")


# 🔹 Ensure compatibility with Windows & Jupyter Notebook by using `if __name__ == "__main__"`
if __name__ == "__main__":
    root_path = "{workspace}/my_project/src/data_origin/video"
    process_videos(root_path)
