import cv2
import os

def main():
    image_folder = 'output'
    video_name = 'output/output_video.mp4'
    fps = 25

    # Get all _vis.jpg files
    images = [img for img in os.listdir(image_folder) if img.endswith("_vis.jpg")]
    
    # Sort them to ensure correct order
    # Assuming filenames are numbers like 000001_vis.jpg
    images.sort(key=lambda x: int(x.split('_')[0]))

    if not images:
        print("No images found in output folder.")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1' or 'XVID'
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    print(f"Creating video {video_name} from {len(images)} images...")

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    print("Video saved successfully!")

if __name__ == "__main__":
    main()
