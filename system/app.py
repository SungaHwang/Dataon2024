import os
import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn.functional as F
import torch.nn as nn

# 경로 설정
UPLOAD_FOLDER = '/mnt/workspace/MyFiles/system/static/uploads'

# 필요한 폴더가 없으면 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 1. VAE 모델 정의 (1000차원 입력을 기대)
class VAE(nn.Module):
    def __init__(self, input_dim=1000, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder_fc1 = nn.Linear(input_dim, 512)
        self.encoder_fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        self.decoder_fc1 = nn.Linear(latent_dim, 256)
        self.decoder_fc2 = nn.Linear(256, 512)
        self.decoder_output = nn.Linear(512, input_dim)

    def encode(self, x):
        h = F.relu(self.encoder_fc1(x))
        h = F.relu(self.encoder_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decoder_fc1(z))
        h = F.relu(self.decoder_fc2(h))
        return torch.sigmoid(self.decoder_output(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

# 2. Adaptive Average Pooling 레이어 추가 (가변 차원 -> 1000차원으로 맞춤)
class FeatureReducer(nn.Module):
    def __init__(self, output_dim=1000):
        super(FeatureReducer, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_dim)  # Adaptive pooling을 사용하여 차원을 맞춤

    def forward(self, x):
        x = x.unsqueeze(0)  # Batch 차원을 추가
        x = self.adaptive_pool(x)
        return x.squeeze(0)  # 다시 Batch 차원 제거

# 3. VAE 모델 불러오기
vae_model_path = 'MyFiles/models/vae_model.pth'
input_dim = 1000  # VAE 입력 차원
latent_dim = 128
vae_model = VAE(input_dim, latent_dim)
vae_model.load_state_dict(torch.load(vae_model_path))  # VAE 가중치 로드
vae_model.eval()

# 4. 차원 축소 모델 (Adaptive Pooling)
feature_reducer = FeatureReducer(output_dim=1000)  # Inception v4의 특징 벡터를 1000차원으로 줄임
feature_reducer.eval()

# 5. 고양이, 강아지 분류 모델 로드
cat_model = timm.create_model('inception_v4', pretrained=False, num_classes=3)
cat_model.load_state_dict(torch.load('/mnt/workspace/MyFiles/models/sample_cat_inception_v4_model.pth'))
cat_model.eval()

dog_model = timm.create_model('inception_v4', pretrained=False, num_classes=5)
dog_model.load_state_dict(torch.load('/mnt/workspace/MyFiles/models/sample_dog_inception_v4_model.pth'))
dog_model.eval()

# 6. 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# 7. Inception v4 모델로부터 특징 벡터 추출 후 차원 축소
def extract_and_reduce_features(image_path, model, feature_reducer):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            features = model.forward_features(img_tensor)  # forward_features로 특징 벡터 추출
        features = features.flatten().cpu()

        # Adaptive Pooling으로 차원을 1000차원으로 줄임
        reduced_features = feature_reducer(features)
        return reduced_features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# 8. VAE 모델 이상 탐지
def detect_anomalies(image_path, vae_model, feature_extractor_model, feature_reducer, threshold=0.1):
    # Inception 모델을 사용하여 이미지 특징 벡터 추출 및 차원 축소
    features = extract_and_reduce_features(image_path, feature_extractor_model, feature_reducer)
    
    if features is None:
        return False
    
    with torch.no_grad():
        recon_features, mu, logvar = vae_model(features)
        recon_error = F.mse_loss(recon_features, features, reduction='mean').item()
    
    return recon_error > threshold  # 임계값 이상이면 이상치로 판단

# 강아지 분류 클래스 레이블 정의
dog_class_labels = {
    0: "비듬_각질_상피성잔고리",
    1: "태선화_과다색소침착",
    2: "농포_여드름",
    3: "미란_궤양",
    4: "결절_종괴"
}

# 고양이 분류 클래스 레이블 정의
cat_class_labels = {
    0: "비듬_각질_상피성잔고리",
    1: "농포_여드름",
    2: "결절_종괴"
}

# 9. 이미지 분류 함수
def classify_image(image_path, model, class_labels):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
    
    predicted_class = preds.item()
    predicted_label = class_labels[predicted_class]  # 숫자 레이블을 클래스명으로 변환
    return predicted_class, predicted_label

# 터미널에서 파일 경로 입력받고 처리하는 함수
def main():
    print("반려동물 피부질환 자가 진단 시스템")
    print("1. 강아지")
    print("2. 고양이")
    animal_choice = input("강아지(1) 또는 고양이(2)를 선택하세요: ")

    if animal_choice not in ['1', '2']:
        print("잘못된 선택입니다. 프로그램을 종료합니다.")
        return

    image_path = input("이미지 파일 경로를 입력하세요: ")

    if not os.path.exists(image_path):
        print("파일이 존재하지 않습니다. 프로그램을 종료합니다.")
        return

    # 1. 이상 탐지 수행
    print("이상 탐지를 수행하고 있습니다...")
    if animal_choice == '1':
        is_anomaly = detect_anomalies(image_path, vae_model, dog_model, feature_reducer)
    else:
        is_anomaly = detect_anomalies(image_path, vae_model, cat_model, feature_reducer)

    if is_anomaly:
        print("이미지에 이상이 감지되었습니다. 다시 이미지를 업로드해주세요.")
    else:
        print("정상 이미지입니다.")

    # 2. 분류 수행
    if animal_choice == '1':
        predicted_class, predicted_label = classify_image(image_path, dog_model, dog_class_labels)
        print(f"강아지 피부질환 분류 결과: 클래스 {predicted_class} ({predicted_label})")
    elif animal_choice == '2':
        predicted_class, predicted_label = classify_image(image_path, cat_model, cat_class_labels)
        print(f"고양이 피부질환 분류 결과: 클래스 {predicted_class} ({predicted_label})")

if __name__ == "__main__":
    main()
