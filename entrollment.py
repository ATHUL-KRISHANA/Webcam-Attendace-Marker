import os
import cv2
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet

mtcnn = MTCNN()
embedder = FaceNet()

data_dir='enrolled_faces'

# Function to preprocess and extract embeddings from a directory of images
def preprocess_and_extract_embeddings(name):
    person_dir=os.path.join(data_dir, name)
    embeddings=[]

    # Loop through all images in the person's folder
    for im_name in os.listdir(person_dir):
        if im_name.endswith('.jpg') or im_name.endswith('.jpeg'):
            image_path=os.path.join(person_dir, im_name)

            image=cv2.imread(image_path)
            rgb_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            detections=mtcnn.detect_faces(rgb_image)

            if detections[0] is not None:
                for i in detections:
                    # print(i)
                    x, y, w, h=i['box']
                    # print(x,y,w,h)
                    face=image[y:y + h, x:x + w]

                    face_resized =cv2.resize(face, (160, 160))

                    embedding = embedder.embeddings([face_resized])[0]
                    embeddings.append(embedding)

    return embeddings

# Save the embeddings to a file
def save_embeddings(embeddings, name):

    pickle.dump(embeddings,open(f'{name}_embeddings.pkl', 'wb'))

for name in os.listdir(data_dir):
    person_dir=os.path.join(data_dir, name)
    pkl_file_path=f'{name}_embeddings.pkl'

    if os.path.isdir(person_dir) and not os.path.exists(pkl_file_path):
        embeddings = preprocess_and_extract_embeddings(name)
        save_embeddings(embeddings, name)
        print(f"Saved embeddings for {name}")
    else:
        print(f"Embeddings already exist for {name}, skipping...")
