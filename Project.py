import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all = True, device = device)
model = InceptionResnetV1(pretrained = 'vggface2').eval().to(device)


def get_face_embeddings(img_path):
    img = Image.open(img_path)
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        faces = mtcnn(img)
        embeddings = model(faces.to(device)).detach().cpu().numpy()
        return embeddings
    else:
        print(f"No face found in {img_path}")
        return None


def calculate_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)


reference_image_paths = ["Photos/Caroline_Wozniacki/Caroline_Wozniacki_1.jpeg",
                         "Photos/Caroline_Wozniacki/Caroline_Wozniacki_2.jpeg",
                         "Photos/Caroline_Wozniacki/Caroline_Wozniacki_3.jpeg",
                         "Photos/Caroline_Wozniacki/Caroline_Wozniacki_4.jpeg",
                         "Photos/Caroline_Wozniacki/Caroline_Wozniacki_5.jpeg",
                         "Photos/Caroline_Wozniacki/Caroline_Wozniacki_6.jpeg",
                         "Photos/Caroline_Wozniacki/Caroline_Wozniacki_7.jpeg",
                         "Photos/King_Diamond/King_Diamond_1.jpg",
                         "Photos/King_Diamond/King_Diamond_2.jpg",
                         "Photos/King_Diamond/King_Diamond_3.jpg",
                         "Photos/King_Diamond/King_Diamond_4.jpg",
                         "Photos/King_Diamond/King_Diamond_5.jpg",
                         "Photos/King_Diamond/King_Diamond_6.jpg",
                         "Photos/King_Diamond/King_Diamond_7.jpg",
                         "Photos/Lars_Ulrich/Lars_Ulrich_1.jpg",
                         "Photos/Lars_Ulrich/Lars_Ulrich_2.jpg",
                         "Photos/Lars_Ulrich/Lars_Ulrich_3.jpg",
                         "Photos/Lars_Ulrich/Lars_Ulrich_4.jpeg",
                         "Photos/Lars_Ulrich/Lars_Ulrich_5.jpg",
                         "Photos/Lars_Ulrich/Lars_Ulrich_6.jpg",
                         "Photos/Lars_Ulrich/Lars_Ulrich_7.jpg",
                         "Photos/Lars_von_Trier/Lars_von_Trier_1.jpg",
                         "Photos/Lars_von_Trier/Lars_von_Trier_2.jpg",
                         "Photos/Lars_von_Trier/Lars_von_Trier_3.jpg",
                         "Photos/Lars_von_Trier/Lars_von_Trier_4.jpg",
                         "Photos/Lars_von_Trier/Lars_von_Trier_5.jpg",
                         "Photos/Lars_von_Trier/Lars_von_Trier_6.jpg",
                         "Photos/Lars_von_Trier/Lars_von_Trier_7.jpg",
                         "Photos/Mads_Mikkelsen/Mads_Mikkelsen_1.jpg",
                         "Photos/Mads_Mikkelsen/Mads_Mikkelsen_2.jpg",
                         "Photos/Mads_Mikkelsen/Mads_Mikkelsen_3.jpg",
                         "Photos/Mads_Mikkelsen/Mads_Mikkelsen_4.jpg",
                         "Photos/Mads_Mikkelsen/Mads_Mikkelsen_5.jpg",
                         "Photos/Mads_Mikkelsen/Mads_Mikkelsen_6.jpg",
                         "Photos/Mads_Mikkelsen/Mads_Mikkelsen_7.jpg"
                         ]

testing_image_paths = ["Photos/Caroline_Wozniacki/Caroline_Wozniacki_8.jpeg",
                       "Photos/Caroline_Wozniacki/Caroline_Wozniacki_9.jpeg",
                       "Photos/King_Diamond/King_Diamond_8.jpg",
                       "Photos/King_Diamond/King_Diamond_9.jpg",
                       "Photos/Lars_Ulrich/Lars_Ulrich_8.jpg",
                       "Photos/Lars_Ulrich/Lars_Ulrich_9.jpg",
                       "Photos/Lars_von_Trier/Lars_von_Trier_8.jpg",
                       "Photos/Lars_von_Trier/Lars_von_Trier_9.jpg",
                       "Photos/Mads_Mikkelsen/Mads_Mikkelsen_8.jpg",
                       "Photos/Mads_Mikkelsen/Mads_Mikkelsen_9.jpg"
                       ]

reference_embeddings = [(get_face_embeddings(img_path), img_path) for img_path in reference_image_paths]

for test_img_path in testing_image_paths:
    test_embedding = get_face_embeddings(test_img_path)
    if test_embedding is not None:
        distances = [(calculate_distance(test_embedding, ref_embedding), ref_img_path) for ref_embedding, ref_img_path in reference_embeddings]
        distances.sort(key = lambda x: x[0])
        closest_images = distances[:3]

        print(f"\nClosest images to {test_img_path}:")
        for dist, img_path in closest_images:
            print(f"Distance: {dist:.3f}, Reference Image Path: {img_path}")