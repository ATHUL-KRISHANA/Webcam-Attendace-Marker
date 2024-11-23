
import cv2
from scipy.spatial.distance import cosine
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os
import pandas as pd
from datetime import datetime

mtcnn = MTCNN()
embedder = FaceNet()

data_dir='enrolled_faces'

log_file='attendance.xlsx'

# Create the Excel file with headers if it doesn't exist
if not os.path.exists(log_file):
    df =pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_excel(log_file, index=False)

def recognize_face(face_embedding):
    min_distance = float('inf')
    recognized_person = None
    
    for name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, name)
        if os.path.isdir(person_dir):
            stored_embeddings=pickle.load(open(f'{name}_embeddings.pkl','rb'))
                
            for stored_embedding in stored_embeddings:
                distance = cosine(face_embedding, stored_embedding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_person = name
                    
    return recognized_person if min_distance < 0.6 else "Unknown"  

# Function to log attendance to Excel
def log_attendance(name):
    current_time = datetime.now()
    current_date = current_time.date()  
    current_time_of_day = current_time.strftime('%H:%M:%S')  
    df = pd.read_excel(log_file)

    # to eliminate the duplicate name entry on the same date
    if df.loc[(df["Name"]==name) & (df["Date"]==str(current_date))].empty:
        new_entry=pd.DataFrame([[name.upper(), current_date, current_time_of_day]], columns=["Name", "Date", "Time"])
        df=pd.concat([df, new_entry], ignore_index=True)

        df["Date"] =pd.to_datetime(df["Date"]).dt.date  
        df["Time"] =pd.to_datetime(df["Time"], format='%H:%M:%S').dt.time  

        df.to_excel(log_file, index=False)
        print(f"Attendance logged for {name.upper()} on {current_date}.")
 

#  video capture
cap =cv2.VideoCapture(0)

while True:
    sucess, frame=cap.read()
    if not sucess:
        break
    
    rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections=mtcnn.detect_faces(rgb_frame)
    
    if len(detections)>0:
        for i in detections:
            x, y, w, h = i['box']
            face =frame[y:y + h, x:x + w]
            
            face_resized =cv2.resize(face, (160, 160))
            face_embedding = embedder.embeddings([face_resized])[0]
            
            person = recognize_face(face_embedding).upper()
            print(f"Recognized: {person}")
            
            if person != "UNKNOWN":
                log_attendance(person)
            
            # cv2.putText(frame, person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Show the video stream
    cv2.imshow("Face Recognition", frame)
    
    # press esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
