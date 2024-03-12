from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolateBallPositions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # converting list into pandas dataframe
        ball_positions_df = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        # interpolate missing values
        ball_positions_df = ball_positions_df.interpolate()
        ball_positions_df = ball_positions_df.bfill()

        # convert back to list
        ball_positions = [{1:x} for x in ball_positions_df.to_numpy().tolist()]

        return ball_positions
    
    def get_ball_shot_frames(self, ball_positions):
        
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # converting list into pandas dataframe
        ball_positions_df = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])
        ball_positions_df['ball_hit'] = 0
        ball_positions_df['mid_y'] = (ball_positions_df['y1'] + ball_positions_df['y2'])/2
        ball_positions_df['mid_y_rolling_mean'] = ball_positions_df['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        ball_positions_df['delta_y'] = ball_positions_df['mid_y_rolling_mean'].diff()
        minimum_change_frames = 25

        for i in range(1, len(ball_positions_df) - int(minimum_change_frames*1.2)):
            negative_position_change = ball_positions_df['delta_y'].iloc[i] > 0 and ball_positions_df['delta_y'].iloc[i+1] < 0
            positive_position_change = ball_positions_df['delta_y'].iloc[i] < 0 and ball_positions_df['delta_y'].iloc[i+1] > 0
            
            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i+1,  i+ int(minimum_change_frames*1.2)+1):
                    negative_position_change_frame = ball_positions_df['delta_y'].iloc[i] > 0 and ball_positions_df['delta_y'].iloc[change_frame] < 0
                    positive_position_change_frame = ball_positions_df['delta_y'].iloc[i] < 0 and ball_positions_df['delta_y'].iloc[change_frame] > 0
                    
                    if negative_position_change and negative_position_change_frame:
                        change_count +=1
                    elif positive_position_change and positive_position_change_frame:
                        change_count +=1
                        
                    if change_count > minimum_change_frames-1:
                        ball_positions_df['ball_hit'].iloc[i] = 1
                        
        frame_with_ball_hits = ball_positions_df[ball_positions_df['ball_hit'] == 1].index.tolist()
        return frame_with_ball_hits

    def detect_frames(self, frames, read_from_stubs=False, stub_path=None):
        ball_detections = []

        if read_from_stubs and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0] # persist tells the models to expect multiple frame inputs
        ball_dict = {}

        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict
    
    def drawBBoxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            #draw bounding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
