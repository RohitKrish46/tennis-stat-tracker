from utils import read_video, save_video
from tracker import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

def main():
    # Read Video
    input_video_path = "input/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    # Detect players and ball
    player_tracker = PlayerTracker(model_path='models/yolov8x')
    ball_tracker = BallTracker(model_path='models/yolov5_last.pt')

    player_detection = player_tracker.detect_frames(video_frames, read_from_stubs=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detection = ball_tracker.detect_frames(video_frames, read_from_stubs=True, stub_path="tracker_stubs/ball_detections.pkl")

    # keypoint mapping
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])
    print(court_keypoints)
    print(len(court_keypoints))
    # Draw output


    # Draw bounding boxes
    output_video_frames = player_tracker.drawBBoxes(video_frames, player_detection)
    output_video_frames = ball_tracker.drawBBoxes(output_video_frames, ball_detection)
    output_video_frames = court_line_detector.drawKeypointsVideo(output_video_frames, court_keypoints)

    # Save Videos
    save_video(output_video_frames, "output/output_video.avi")

if __name__== "__main__":
    main()
