"""Script to extract a thumbnail frame from the video."""
import sys
import os

def create_thumbnail(video_path, output_path, frame_time=2):
    """
    Extract a frame from video to create a thumbnail.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the thumbnail image
        frame_time: Time in seconds to extract frame (default: 2 seconds)
    """
    try:
        import imageio
        from PIL import Image
    except ImportError:
        print("Installing required packages (imageio, Pillow)...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio", "imageio-ffmpeg", "Pillow", "-q"])
            import imageio
            from PIL import Image
        except Exception as e:
            print(f"Error installing packages: {e}")
            print("\nPlease install manually: pip install imageio imageio-ffmpeg Pillow")
            return False
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    try:
        # Read video
        reader = imageio.get_reader(video_path)
        metadata = reader.get_meta_data()
        
        # Get video properties
        fps = metadata.get('fps', 30)
        duration = metadata.get('duration', 0)
        
        # Calculate frame number (use 2 seconds or 10% of video, whichever is smaller)
        if duration > 0:
            target_time = min(frame_time, duration * 0.1)
        else:
            # If duration not available, estimate from frame count
            try:
                total_frames = reader.count_frames()
                target_time = min(frame_time, (total_frames / fps) * 0.1) if fps > 0 else frame_time
            except:
                target_time = frame_time
        
        frame_number = int(target_time * fps) if fps > 0 else 0
        
        # Read the frame
        reader.set_image_index(frame_number)
        frame = reader.get_next_data()
        
        # Convert to PIL Image
        img = Image.fromarray(frame)
        
        # Resize to reasonable thumbnail size (1280x720 max, maintain aspect ratio)
        max_width = 1280
        max_height = 720
        
        if img.width > max_width or img.height > max_height:
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        
        # Save thumbnail
        img.save(output_path, 'JPEG', quality=90)
        print(f"✓ Thumbnail created: {output_path}")
        print(f"  Size: {img.width}x{img.height} pixels")
        reader.close()
        return True
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return False

if __name__ == "__main__":
    video_file = "Article_forge_recording.mp4"
    thumbnail_file = "Article_forge_recording_thumbnail.jpg"
    
    print(f"Creating thumbnail from {video_file}...")
    success = create_thumbnail(video_file, thumbnail_file)
    
    if success:
        print(f"\n✓ Success! Thumbnail saved as: {thumbnail_file}")
    else:
        print("\n✗ Failed to create thumbnail")
        sys.exit(1)
