import numpy as np
import torch
import dlib
from torch import nn
from sklearn.metrics import pairwise_distances


class FaceRecognitionEstimator:
    def __init__(self):
        self.model = SiameseNetwork()
        self.model.load_state_dict(torch.load("model.pth",
                                              map_location=torch.device('cpu')))
        self.model.eval()
        self.embeddings = np.load("emb_all.npy")
        self.id_s = np.load("id_all.npy")
        self.detector = dlib.get_frontal_face_detector()

    def predict(self, image):
        faces = self.detector(np.array(image), 1)
        for i, d in enumerate(faces):
            image = image.crop((d.left(), d.top(), d.right(), d.bottom())).resize((100, 100))

            img_embedding = self.model(
                torch.Tensor(
                    np.array(image) / 255
                ).permute(2, 0, 1).unsqueeze(0)
            ).detach().numpy()
            distances = pairwise_distances(self.embeddings,
                                       img_embedding).reshape(-1)

            min_3 = np.argsort(distances)[:3]
            str_list = list(self.id_s[min_3].reshape(-1))
            return list(map(int, str_list))




class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3)

        )

        self.fc1 = nn.Sequential(
            nn.Linear(3136, 128)
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        #print(output.shape)
        output = self.fc1(output)
        return output