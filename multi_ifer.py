import cv2
import torch
from torchvision import transforms

from utils import model_select

# Configuração do dispositivo

MODEL_NAME1 = "VGG16"  # EfficientNet, DaViT, VGG16, Resnet50 or MobileNet
MODEL_PATH1 = fr"results\best_{MODEL_NAME1}.pth"
MODEL_NAME2 = "MobileNet"  # EfficientNet, DaViT, VGG16, Resnet50 or MobileNet
MODEL_PATH2 = fr"results\best_{MODEL_NAME2}.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model1 = model_select(MODEL_NAME1)
model1.load_state_dict(torch.load(MODEL_PATH1))
model1.to(DEVICE)
model1.eval()

model2 = model_select(MODEL_NAME2)
model2.load_state_dict(torch.load(MODEL_PATH2))
model2.to(DEVICE)
model2.eval()

# Transformação para o frame da câmera
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

expres_dict = { 0:'raiva',
                1:'desprezo',
                2:'nojo',
                3:'medo',
                4:'alegria',
                5:'neutro',
                6:'triste',
                7:'surpresa'}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para escala de cinza para a detecção de rostos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 8)

    scale_factor = 0.5

    for (x, y, w, h) in faces:
        # Calcular a nova área de detecção com margem extra
        x_new = int(x - w * scale_factor / 2)
        y_new = int(y - h * scale_factor / 2)
        w_new = int(w * (1 + scale_factor))
        h_new = int(h * (1 + scale_factor))

        # Garantir que os novos valores não saiam fora da imagem
        x_new = max(x_new, 0)
        y_new = max(y_new, 0)
        w_new = min(w_new, frame.shape[1] - x_new)
        h_new = min(h_new, frame.shape[0] - y_new)

        # Desenhar um retângulo ao redor do rosto (opcional)
        cv2.rectangle(frame, (x_new, y_new), (x_new+w_new, y_new+h_new), (255, 0, 0), 2)
        
        # Ajustar o ROI para o rosto detectado
        roi_color = frame[y:y_new+h_new, x:x_new+w_new]
        
        # Transformar o ROI antes da inferência
        try:
            # Certifique-se de que o ROI não está vazio
            if roi_color.size != 0:
                roi_transformed = transform(roi_color).unsqueeze(0).to(DEVICE)

                # Realizar a inferência no ROI
                with torch.no_grad():
                    outputs1 = model1(roi_transformed)
                    outputs2 = model2(roi_transformed)
                    outputs_mean = (outputs1 + outputs2) / 2
                    _, predicted = torch.max(outputs_mean, 1)
                    label = f"Prediction: {expres_dict[predicted.item()]}"  # Substitua pelo seu próprio mapeamento de etiquetas, se necessário

                # Adicionar texto com a predição no ROI
                cv2.putText(frame, label, (x_new, y_new-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        except Exception as e:
            print(f"Erro durante o processamento do ROI: {e}")

    # Mostrar o frame
    cv2.imshow('Webcam', frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()