import cv2
import numpy as np

def decrease_brightness(input_video_path, output_video_path, brightness_factor):
    """
    Decrease the brightness of a video.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file.
        brightness_factor (float): Factor to decrease brightness (0.0 to 1.0).
                                   1.0 means no change, 0.0 means completely dark.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files 

    # Create VideoWriter object to save the output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Decrease brightness by multiplying the frame with the brightness factor
        darkened_frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)

        # Write the processed frame to the output video
        out.write(darkened_frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Video saved to {output_video_path}")

# Example usage
input_video_path = "videos/dark_condition.avi"  # Path to your input video
output_video_path = "videos/darker.mp4"  # Path to save the output video
brightness_factor = 0.5  # Reduce brightness to 50%

decrease_brightness(input_video_path, output_video_path, brightness_factor)