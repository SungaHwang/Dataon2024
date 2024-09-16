import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import timm  # Inception v4 모델을 로드할 수 있는 라이브러리
from torchvision import transforms
from PIL import Image
import glob
import pickle
import random

# 1. 저장된 Inception v4 모델 로드
model_path = 'MyFiles/models/50000_AnomalyDetection_inception_v4__model.pth'
model = timm.create_model('inception_v4', pretrained=False)
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"Inception v4 모델이 '{model_path}'에서 로드되었습니다.")

# 2. 이미지 전처리 함수 (Inception v4에 맞게)
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 3. 이미지 특징 추출 함수
def extract_features(image_path, model):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            features = model(img_tensor)
        features = features.flatten().cpu().numpy()
        return features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# 4. VAE 모델 정의
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
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

# 5. 저장된 VAE 모델 불러오기
vae_model_path = 'MyFiles/models/vae_model.pth'
input_dim = 1000  # 저장된 모델의 입력 차원과 동일하게 수정
latent_dim = 128
vae = VAE(input_dim, latent_dim)
vae.load_state_dict(torch.load(vae_model_path))
vae.eval()
print(f"VAE 모델이 '{vae_model_path}'에서 로드되었습니다.")

# 6. 이상치 탐지 함수
def detect_anomalies_vae(image_features, vae, threshold=0.1):
    vae.eval()
    with torch.no_grad():
        image_tensor = torch.tensor(image_features, dtype=torch.float32).unsqueeze(0)  # 배치 차원 추가
        recon_image, _, _ = vae(image_tensor)
        recon_error = F.mse_loss(recon_image, image_tensor, reduction='mean').item()
        return recon_error > threshold  # 재건 오차가 threshold 이상이면 이상치로 간주

# 7. 새로운 이미지에 대한 이상치 탐지
new_image_paths = glob.glob('MyFiles/data/anomalydetection/*.jpg')  # 테스트할 이미지 경로들
threshold = 0.1  # 임계값 (조정 가능)

# 결과 저장을 위한 경로 설정
results_file = 'MyFiles/results/vae_anomaly_detection_results.txt'

with open(results_file, 'w') as f:
    for new_image_path in tqdm(new_image_paths, desc="Testing Images"):
        new_image_features = extract_features(new_image_path, model)
        if new_image_features is not None:
            is_anomaly = detect_anomalies_vae(new_image_features, vae, threshold)
            if is_anomaly:
                result_str = f"{new_image_path}는 이상치입니다.\n"
            else:
                result_str = f"{new_image_path}는 정상 이미지입니다.\n"
            f.write(result_str)  # 결과를 파일에 저장
            print(result_str.strip())

print(f"결과가 '{results_file}'에 저장되었습니다.")
