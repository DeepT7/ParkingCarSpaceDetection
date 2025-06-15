import cv2
import os 

def extract_frames(video_path, output_dir, frame_numbers):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video ")
        return 
    os.makedirs(output_dir, exist_ok=True)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames in the video: {frame_count}")

    for frame_number in frame_numbers:
        if frame_number >= frame_count:
            continue 

        # Set the video to the desired frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret: 
            frame_path = os.path.join(output_dir, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Extracted frame {frame_number} to {frame_path}")
        else:
            print(f"Error: Couldn't read the frame {frame_number}")

    cap.release()
    print("Frame extraction completed")

video_path = 'videos/darker.mp4'
output_dir = 'frames'
frame_numbers = [0, 50, 100, 150, 200]

extract_frames(video_path, output_dir, frame_numbers)
