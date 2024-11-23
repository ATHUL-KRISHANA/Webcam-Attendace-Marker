# Webcam Attendance Marker

## Overview
This system uses a webcam to mark attendance by recognizing faces in real time. It consists of two main scripts:

- **`enrollment.py`**: Creates and stores face embeddings for each person.
- **`attendance_marker.py`**: Marks attendance by recognizing faces and logs it in an Excel file.

## Libraries Used
- **MTCNN**: Used for face detection.
- **FaceNet**: Used for recognizing and generating face embeddings.

## How to Run

### Step 1: Prepare the Directories
1. **Create a directory**: In the folder where you want to run the system, create a directory named `enrolled_faces`.
2. **Create subdirectories**: Inside the `enrolled_faces` directory, create a separate folder for each person (e.g., `John_Doe`). 
   - The name of each folder will be saved in the `attendance.xlsx` file as the attendance entry.

### Step 2: Enrollment Process
1. **Run the `enrollment.py` script**: This script generates and saves the face embeddings (512 numerical points) for each person in the `enrolled_faces` directory.
   - The embeddings are stored in a pickle file, which contains the face features for each enrolled person.
   -   ```bash
   python3 enrollment.py
### Step 3: Running the Attendance Marke
1. **Run the `attendance_marker.py` script**: Once the embeddings are generated, you can run the attendance marker.
- It will automatically detect faces using your webcam, recognize them based on the saved embeddings, and log the attendance in the `attendance.xlsx` file.

   ```bash
   python3 attendace_marker.py
## File Structure:
Your project folder should be organized as follows:
```
├── enrolled_faces/     # Folder to store enrolled faces (one subfolder per person)
│   └── <person_name>/  # Example: Athul/, Karthikeyan/ (created during enrollment)
├── attendance.xlsx     # Excel file for storing attendance records (created automatically)
├── requirements.txt    # Text file listing project dependencies (optional)
└── enrollment.py   
```

```
