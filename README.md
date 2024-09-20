# tennis-stat-tracker

This project aims to automatically compute match statistics from tennis footage by extracting key metrics such as player movement speed, shot velocity, and shot count, etc. which can be further leveraged for research and analysis. We employ YOLOv8 for real-time detection of players and the tennis ball, while CNNs are utilized to identify key court landmarks. This allows us to create a scaled representation of the court for accurate ball tracking and detailed performance analysis.

## Output

![Final Output](./output/output_video.gif)