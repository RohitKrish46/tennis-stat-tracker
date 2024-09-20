# Tennis Stat Tracker

This project aims to automatically compute match statistics from tennis gameplay footage by extracting key metrics such as player movement speed, shot velocity, and shot count, etc. which can be further leveraged for research and analysis. We employ YOLOv8 for real-time detection of players and the tennis ball, while CNNs are utilized to identify key court landmarks. This allows us to create a scaled representation of the court for accurate ball tracking and detailed performance analysis.


## Datasets

1. **Tennis Ball Detection Dataset** - This dataset contains a total of 578 images, split into three sets: 74% for training (428 images), 9% for testing (50 images), and 17% for validation (100 images). Each image includes a single annotation of a tennis ball, with a resolution of 1280×720. You can access the dataset from [here](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection)

2. **Tennis Court Keypoint Detection Dataset** - This dataset contains 8,841 images, divided into a 75% training set and a 25% validation set. Each image includes 14 annotated keypoints, with a resolution of 1280×720. The dataset covers all court types: hard, clay, and grass. You can download the dataset [here](https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view?usp=drive_link). For more details on how the dataset was collected for keypoint detection, visit the original repository [here](https://github.com/yastrebksv/TennisCourtDetector). 


## Models Used

1. Player Detection - [YOLOv8](https://github.com/ultralytics/ultralytics)

2. Ball Detection - [YOLOv5](https://github.com/ultralytics/yolov5)

3. Court Keypoint detection - [Finetuned ResNet50](https://blog.roboflow.com/what-is-resnet-50/)


## Workflow

1. **Player Detection Using YOLOv8** - We begin by experimenting with YOLOv8 for general object detection in images, focusing on isolating specific objects—in this case, the players and the tennis ball.

2. **Fine-Tuning YOLOv5 for Tennis Ball Detection** - After discovering that YOLOv8 is less effective for detecting tennis balls, we shift to YOLOv5, selected due to dataset compatibility constraints. We then fine-tune YOLOv5 using a tennis ball detection dataset sourced from Roboflow.

3. **Keypoint Detection Using ResNet50** - To accurately map the court keypoints in each frame, we leveraged a publicly available [dataset]([dataset](https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view?usp=drive_link)) and fine-tuned a ResNet50 model. ResNet50 was chosen for its proven depth and efficiency in image classification, making it well-suited for this task.

4. **Tennis Analyzer Workflow** - Next, we implemented the workflow in the main.py file. This script orchestrates the three models to process each frame of the input_video, outputting frames enriched with the detected keypoints and other relevant information.

5. **Mini-Court Generation with Game Statistics** - Lastly, as each frame is processed, a mini-court is rendered on the right side of the screen. This visual aid allows viewers to track the game in real-time while displaying key statistics such as player movement speed, shot velocity, shot count, and more, offering a comprehensive view of the match.


## Output

Here is a sample output after processing an entire video:

![Final Output](./output/output_video.gif)


## References

1. Big thanks to the walkthrough on AI/ML Tennis Analysis system with YOLO, PyTorch, and Key Point Extraction [Code In a Jiffy](https://www.youtube.com/watch?v=L23oIHZE14w)

2. Thanks to [Court Keypoint detection](https://github.com/yastrebksv/TennisCourtDetector) repo for the dataset to train court keypoints.

3. [YOLOv8](https://github.com/ultralytics/ultralytics)

4. More about [ResNet50](https://blog.roboflow.com/what-is-resnet-50/)