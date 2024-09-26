import os
from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn.functional as F
import torch.nn as nn

app = Flask(__name__)

# 파일 저장 경로
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 1. 고양이, 강아지 모델 불러오기
cat_model = timm.create_model('inception_v4', pretrained=False, num_classes=3)
cat_model.load_state_dict(torch.load('model/sample_cat_inception_v4_model.pth', map_location=torch.device('cpu')))
cat_model.eval()

dog_model = timm.create_model('inception_v4', pretrained=False, num_classes=5)
dog_model.load_state_dict(torch.load('model/sample_dog_inception_v4_model.pth', map_location=torch.device('cpu')))
dog_model.eval()

# 2. 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# 3. 이상 탐지 모델 정의 (간단하게 MSE 기준으로 탐지)
class SimpleVAE(nn.Module):
    def __init__(self, input_dim=1000, latent_dim=128):
        super(SimpleVAE, self).__init__()
        self.encoder_fc = nn.Linear(input_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = F.relu(self.encoder_fc(x))
        return torch.sigmoid(self.decoder_fc(z))

# vae_model = SimpleVAE()
# vae_model.load_state_dict(torch.load('vae_model.pth'))
# vae_model.eval()

# # 4. 이상 탐지 수행
# def detect_anomaly(image_tensor):
#     with torch.no_grad():
#         recon = vae_model(image_tensor)
#         mse_loss = F.mse_loss(recon, image_tensor)
#     return mse_loss.item() > 0.1

# 5. 이미지 분류 함수
def classify_image(image_tensor, model, class_labels):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    return class_labels[preds.item()]

# 5. 이미지 분류 함수
def classify_image(image_tensor, model, class_labels):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    return class_labels[preds.item()]

# 6. 분류 클래스 정의
dog_class_labels = {
    0: "비듬_각질_상피성잔고리",
    1: "태선화_과다색소침착",
    2: "농포_여드름",
    3: "미란_궤양",
    4: "결절_종괴"
}

cat_class_labels = {
    0: "비듬_각질_상피성잔고리",
    1: "농포_여드름",
    2: "결절_종괴"
}

# 7. 분석 요청 처리
# 파일 업로드 및 분석 처리 엔드포인트
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다.'})
    
    file = request.files['file']
    animal_type = request.form.get('animalType')
    
    if not animal_type:
        return jsonify({'error': '동물 타입이 선택되지 않았습니다.'})

    # 파일 저장
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # 이미지 로드 및 전처리
    try:
        img = Image.open(file_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)
    except Exception as e:
        return jsonify({'error': f'이미지 처리 중 오류 발생: {str(e)}'})

    # 이상 탐지
    # if detect_anomaly(img_tensor):
    #     return jsonify({'isAnomaly': True})

    # 동물에 따른 분류
    if animal_type == 'dog':
        disease = classify_image(img_tensor, dog_model, dog_class_labels)
    elif animal_type == 'cat':
        disease = classify_image(img_tensor, cat_model, cat_class_labels)
    else:
        return jsonify({'error': '유효하지 않은 동물 타입입니다.'})

    return jsonify({'isAnomaly': False, 'disease': disease})

if __name__ == '__main__':
    app.run(debug=True)