import cv2
from constants.config_constants import DOUBLE_LINE_WIDTH
import pandas as pd
from utils import read_video, save_video, measure_distance, convert_pixel_distance_to_meters, draw_player_stats
from tracker import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_map import MiniCourt
from copy import deepcopy

def main():
    # Read Video
    input_video_path = "input/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    # Detect players and ball
    player_tracker = PlayerTracker(model_path='models/yolov8x')
    ball_tracker = BallTracker(model_path='models/yolov5_last.pt')

    player_detection = player_tracker.detect_frames(video_frames, read_from_stubs=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detection = ball_tracker.detect_frames(video_frames, read_from_stubs=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detection = ball_tracker.interpolateBallPositions(ball_detection)
    
    # keypoint mapping
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    player_detection = player_tracker.chooseAndFilterPlayers(court_keypoints, player_detection)

    # Mini-Map
    mini_map = MiniCourt(video_frames[0])

    # detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detection)
    # print(ball_shot_frames)

    # convert real positions to minimap positions
    player_mini_map_detections, ball_mini_map_detections = mini_map.convert_bounding_boxes_to_mini_map_coordinates(player_detection, ball_detection, court_keypoints)

    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]
    
    for ball_shot_ind in range(len(ball_shot_frames) -1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_sec = (end_frame - start_frame)/24

        # get distance covered
        distance_covered_by_ball_in_pixels = measure_distance(ball_mini_map_detections[start_frame][1], 
                                                              ball_mini_map_detections[end_frame][1])
        distance_covered_by_ball_in_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_in_pixels,
                                                                              DOUBLE_LINE_WIDTH,
                                                                              mini_map.get_width_of_mini_court())

        # speed of ball shot
        speed_of_ball = distance_covered_by_ball_in_meters/ball_shot_time_in_sec * 3.6
        
        # Player who shot ball
        player_positions = player_mini_map_detections[start_frame]
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                               ball_detection[start_frame][1]))

        # player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_map_detections[start_frame][opponent_player_id], 
                                                              player_mini_map_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels,
                                                                               DOUBLE_LINE_WIDTH,
                                                                               mini_map.get_width_of_mini_court())

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_sec * 3.6

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']



    # Draw bounding boxes
    output_video_frames = player_tracker.drawBBoxes(video_frames, player_detection)
    output_video_frames = ball_tracker.drawBBoxes(output_video_frames, ball_detection)

    # draw court keypoints
    output_video_frames = court_line_detector.drawKeypointsVideo(output_video_frames, court_keypoints)

    # Draw mini-map
    output_video_frames = mini_map.draw_mini_court(output_video_frames)
    output_video_frames = mini_map.draw_points_on_mini_court(output_video_frames, player_mini_map_detections)
    output_video_frames = mini_map.draw_points_on_mini_court(output_video_frames, ball_mini_map_detections, (0, 255, 255))
    
    # draw player stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # draw FPS
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame:{i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save Videos
    save_video(output_video_frames, "output/output_video.avi")

if __name__== "__main__":
    main()
