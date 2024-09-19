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
Image.LOAD_TRUNCATED_IMAGES = True

# Inception v4 모델 로드 (사전 학습된 가중치 사용)
model = timm.create_model('inception_v4', pretrained=True)
model.eval()  # 평가 모드로 전환

# 이미지 전처리 함수 (Inception v4에 맞게)
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def extract_features(image_path, model):
    try:
        img = Image.open(image_path).convert('RGB')  # 이미지를 RGB로 변환
        img_tensor = preprocess(img).unsqueeze(0)  # 배치 차원을 추가

        with torch.no_grad():  # 추론 모드에서 그라디언트 계산 방지
            features = model(img_tensor)  # forward()로 변경
        features = features.flatten().cpu().numpy()  # numpy 배열로 변환
        return features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# 모든 이미지 경로 가져오기
dog_cat_image_paths = glob.glob('MyFiles/data/train_new/**/*.[jJpP][pPnN][gG]', recursive=True)

# 사용하고 싶은 이미지 개수 (랜덤하게 선택)
sample_size = 50000  # 사용할 이미지 개수
if sample_size < len(dog_cat_image_paths):
    dog_cat_image_paths = random.sample(dog_cat_image_paths, sample_size)

print(f"{sample_size}개의 이미지를 랜덤으로 선택했습니다.")

# 이미지로부터 특징 벡터 추출
dog_cat_features = np.array([feat for feat in [extract_features(img_path, model) for img_path in tqdm(dog_cat_image_paths, desc="Processing Images")] if feat is not None])

# 모델 저장
model_path = 'MyFiles/models/50000_AnomalyDetection_inception_v4__model.pth'
torch.save(model.state_dict(), model_path)
print(f"inception모델이 '{model_path}'에 저장되었습니다.")

# 저장된 Inception v4 모델 로드
model_path = 'MyFiles/models/50000_AnomalyDetection_inception_v4__model.pth'
model = timm.create_model('inception_v4', pretrained=False)  # 사전 학습된 가중치는 사용하지 않음
model.load_state_dict(torch.load(model_path))
model.eval()  # 평가 모드로 전환
print(f"Inception v4 모델이 '{model_path}'에서 로드되었습니다.")

# 이미지 전처리 함수 (Inception v4에 맞게)
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # ImageNet에 맞게 정규화
])

# 이미지 특징 추출 함수
def extract_features(image_path, model):
    try:
        img = Image.open(image_path).convert('RGB')  # 이미지를 RGB로 변환
        img_tensor = preprocess(img).unsqueeze(0)  # 배치 차원을 추가

        with torch.no_grad():  # 추론 모드에서 그라디언트 계산 방지
            features = model(img_tensor)
        features = features.flatten().cpu().numpy()  # numpy 배열로 변환
        return features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# VAE 모델 정의
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

# 5. VAE 학습 함수
def train_vae(vae, data, epochs=10, batch_size=64, learning_rate=1e-3):
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    vae.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for i in tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            loss = vae.loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(data)}")

# 강아지와 고양이 이미지에서 특징 추출
dog_cat_image_paths = glob.glob('MyFiles/data/train_new/**/*.[jJpP][pPnN][gG]', recursive=True)

# 사용하고 싶은 이미지 개수 (랜덤하게 선택)
sample_size = 50000  # 사용할 이미지 개수
if sample_size < len(dog_cat_image_paths):
    dog_cat_image_paths = random.sample(dog_cat_image_paths, sample_size)

print(f"{sample_size}개의 이미지를 랜덤으로 선택했습니다.")

# 이미지로부터 특징 벡터 추출
dog_cat_features = np.array([feat for feat in [extract_features(img_path, model) for img_path in tqdm(dog_cat_image_paths, desc="Processing Images")] if feat is not None])

# VAE 모델 인스턴스 생성
input_dim = dog_cat_features.shape[1]  # 특징 벡터 차원 수
latent_dim = 128  # 잠재 공간 차원 수
vae = VAE(input_dim, latent_dim)

#  VAE 학습
train_vae(vae, dog_cat_features, epochs=50, batch_size=64, learning_rate=1e-3)

#  VAE 모델 저장
vae_model_path = 'MyFiles/models/vae_model.pth'
torch.save(vae.state_dict(), vae_model_path)
print(f"VAE 모델이 '{vae_model_path}'에 저장되었습니다.")

#  저장된 VAE 모델 불러오기
vae = VAE(input_dim, latent_dim)  # VAE 모델 인스턴스를 새로 생성
vae.load_state_dict(torch.load(vae_model_path))
vae.eval()  # 평가 모드로 전환
print(f"VAE 모델이 '{vae_model_path}'에서 로드되었습니다.")

# 여러 이미지에 대한 이상치 탐지 (재건 오차 기반)
def detect_anomalies_vae(image_features, vae, threshold=0.1):
    vae.eval()
    with torch.no_grad():
        image_tensor = torch.tensor(image_features, dtype=torch.float32)
        recon_image, _, _ = vae(image_tensor)
        recon_error = F.mse_loss(recon_image, image_tensor, reduction='mean').item()
        return recon_error > threshold  # 재건 오차가 threshold 이상이면 이상치로 간주

# 여러 이미지에 대한 이상치 탐지 실행
new_image_paths = glob.glob('MyFiles/data/anomalydetection/*.jpg')  # 테스트할 이미지 경로들
threshold = 0.1  # 임계값 (조정 가능)

for new_image_path in tqdm(new_image_paths, desc="Testing Images"):
    new_image_features = extract_features(new_image_path, model)
    if new_image_features is not None:
        is_anomaly = detect_anomalies_vae(new_image_features, vae, threshold)
        if is_anomaly:
            print(f"{new_image_path}는 이상치입니다.")
        else:
            print(f"{new_image_path}는 정상 이미지입니다.")
