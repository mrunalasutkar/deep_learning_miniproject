import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms

# DEVICE

device = torch.device(

    "cuda" if torch.cuda.is_available()

    else "cpu"
)

# IMAGE TRANSFORM

transform = transforms.Compose([

    transforms.Resize((128,128)),

    transforms.ToTensor()
])

# CNN MODEL

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3,16,3),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64*14*14,512),

            nn.ReLU(),

            nn.Linear(512,1),

            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.conv(x)

        x = self.fc(x)

        return x

# LOAD MODEL

model = CNN().to(device)

model.load_state_dict(

    torch.load(

        'model/model.pth',

        map_location=device
    )
)

model.eval()

# LOAD TEST IMAGE

image_path = 'test.jpg'


image = Image.open(image_path).convert('RGB')


image = transform(image)


image = image.unsqueeze(0).to(device)

# PREDICTION

with torch.no_grad():

    output = model(image)

    prediction = (output > 0.5).float()

# RESULT

if prediction.item() == 1:

    print("Dog")

else:

    print("Cat")