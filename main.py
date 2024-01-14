'''
All the code was developed by Flávio Moura, a scientific research from the Federal University of Pará and member of the 
Operational Research Laboratory. This repo aims to train multiple models for the task of facial expression recognition.
The main objetive of the development of that models is to be deployed in monitoring tools from the UX-Tracking Framework,
also developed by Flávio Moura. For that reason, the models developed was evaluated focusing on Precision metric and low
computational cost architectures of AI models was selected.

For more informations about:
    - Development of the models
    - Architecture of the UX-Tracking framework
    - Scientif Research behind this code and the fully framework
    - Colaboration in a scientific research

Please, contact Flávio Moura in the email: flavio.moura@itec.ufpa.br
'''

import torch

from utils import load_data, model_select, train_model

# Configurações principais
data_dir = 'AffectNet'
img_height, img_width = 96, 96
batch_size = 512
epochs = 100

train_loader, val_loader = load_data(data_dir, img_height, img_width, batch_size)
model = model_select(img_height, img_width)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_model(model, train_loader, val_loader, device, epochs)