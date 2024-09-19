import os
import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import glob

# 강아지와 고양이 질병 CSV 파일 경로 (home 디렉토리 기반)
home_dir = os.path.expanduser('~')
dog_diseases_csv = os.path.join(home_dir,  'workspace', 'MyFiles', 'results', 'Mistral-7B_dog_classified_diseases.csv')
cat_diseases_csv = os.path.join(home_dir,  'workspace', 'MyFiles', 'results', 'Mistral-7B_cat_classified_diseases.csv')

dog_disease_details_csv = os.path.join(home_dir, 'workspace', 'MyFiles', 'data','csv_output', 'dog_skin_diseases.csv')
cat_disease_details_csv = os.path.join(home_dir, 'workspace', 'MyFiles','data','csv_output', 'cat_skin_diseases.csv')

# 경로 설정
UPLOAD_FOLDER = os.path.join(home_dir, 'workspace', 'MyFiles','system', 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 6. 강아지 및 고양이 분류 레이블
dog_class_labels = {
    0: "비듬_각질_상피성잔고리 (Dandruff_Keratin_Epithelial Colloid)",
    1: "태선화_과다색소침착 (Lichenification_Hyperpigmentation)",
    2: "농포_여드름 (Pustules_Acne)",
    3: "미란_궤양 (Erosion_Ulcer)",
    4: "결절_종괴 (Nodule_Tumor)"
}

cat_class_labels = {
    0: "비듬_각질_상피성잔고리 (Dandruff_Keratin_Epithelial Colloid)",
    1: "농포_여드름 (Pustules_Acne)",
    2: "결절_종괴 (Nodule_Tumor)"
}

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


# 2. 저장된 Inception v4 모델 로드 및 VAE 모델 로드
model_path = os.path.join(home_dir,  'workspace', 'MyFiles', 'models', '50000_AnomalyDetection_inception_v4__model.pth')
model = timm.create_model('inception_v4', pretrained=False)
model.load_state_dict(torch.load(model_path))
model.eval()

vae_model_path = os.path.join(home_dir,  'workspace', 'MyFiles','models', 'vae_model.pth')
input_dim = 1000
latent_dim = 128
vae = VAE(input_dim, latent_dim)
vae.load_state_dict(torch.load(vae_model_path))
vae.eval()

cat_model_path = os.path.join(home_dir,  'workspace', 'MyFiles','models', 'classification_cat_inception_v4_model.pth')
dog_model_path = os.path.join(home_dir,  'workspace', 'MyFiles','models', 'classification_dog_inception_v4_model.pth')

cat_model = timm.create_model('inception_v4', pretrained=False, num_classes=3)
cat_model.load_state_dict(torch.load(cat_model_path))
cat_model.eval()

dog_model = timm.create_model('inception_v4', pretrained=False, num_classes=5)
dog_model.load_state_dict(torch.load(dog_model_path))
dog_model.eval()

# 3. 이미지 전처리 함수 (Inception v4에 맞게)
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 4. 이미지 특징 추출 함수
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

# 5. 이상 탐지 함수
def detect_anomalies(image_features, vae_model, threshold=1.3):
    vae_model.eval()
    with torch.no_grad():
        image_tensor = torch.tensor(image_features, dtype=torch.float32).unsqueeze(0)
        recon_image, _, _ = vae_model(image_tensor)
        recon_error = F.mse_loss(recon_image, image_tensor, reduction='mean').item()
        #print(recon_error)
        return recon_error > threshold  # 재건 오차가 threshold 이상이면 이상치로 간주


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


# 8. 세부 질병 리스트 확인
def view_disease_list(animal_choice):
    if animal_choice == '1':
        df = pd.read_csv(dog_diseases_csv)
    else:
        df = pd.read_csv(cat_diseases_csv)
    
    # 질병명을 가나다 순으로 정렬
    df_sorted = df.sort_values(by='disease_name')
    print("\n[세부 질병 목록]")
    for index, row in df_sorted.iterrows():
        print(f"- {row['disease_name']}")  # 질병 목록을 이름으로 표시

    while True:
        selected_disease_name = input("\n확인하고 싶은 세부 질병 이름을 입력하세요 (종료하려면 'exit' 입력): ")

        if selected_disease_name.lower() == 'exit':
            print("프로그램을 종료합니다.")
            return None

        # 사용자가 입력한 질병 이름이 목록에 있는지 확인
        if selected_disease_name in df_sorted['disease_name'].values:
            return selected_disease_name
        else:
            print("잘못된 질병 이름입니다. 다시 입력해주세요.")

# 9. 세부 질병 정보 확인 함수
def show_disease_details(selected_disease_name, animal_choice):
    if animal_choice == '1':
        df_details = pd.read_csv(dog_disease_details_csv)
    else:
        df_details = pd.read_csv(cat_disease_details_csv)
    
    disease_info = df_details[df_details['disease_name'] == selected_disease_name]
    if disease_info.empty:
        print(f"{selected_disease_name}에 대한 정보를 찾을 수 없습니다.")
        return

    print(f"\n[세부 정보: {selected_disease_name}]")
    print(f"정의 (define): {disease_info['define'].values[0]}")
    print(f"원인 (Cause): {disease_info['cause'].values[0]}")
    print(f"증상 (Condition): {disease_info['condition'].values[0]}")
    print(f"치료법 (Treatment): {disease_info['treatment'].values[0]}")

# 10. 메인 함수
def main():
    while True:
        print("\n반려동물 피부질환 자가 진단 시스템")
        print("1. 강아지")
        print("2. 고양이")
        animal_choice = input("강아지(1) 또는 고양이(2)를 선택하세요 (프로그램 종료를 원하면 'exit'을 입력하세요): ")

        if animal_choice.lower() == 'exit':
            print("프로그램을 종료합니다.")
            break

        if animal_choice not in ['1', '2']:
            print("잘못된 선택입니다. 다시 시도하세요.")
            continue

        while True:  # 이미지 업로드를 반복할 수 있도록 반복문 추가
            image_path = input("이미지 파일 경로를 입력하세요: ")

            if not os.path.exists(image_path):
                print("파일이 존재하지 않습니다. 다시 시도하세요.")
                continue

            # 1. 이상 탐지 수행
            print("이상 탐지를 수행하고 있습니다...")
            image_features = extract_features(image_path, model)
            if image_features is not None:
                is_anomaly = detect_anomalies(image_features, vae)
                if is_anomaly:
                    print("이미지에 이상이 감지되었습니다. 다시 이미지를 업로드해주세요.")
                    continue  # 이상이 감지되면 다시 이미지 업로드로 돌아가도록 continue
                else:
                    print("정상 이미지입니다.")
                    break  # 정상 이미지일 경우 반복문 탈출

        # 2. 분류 수행
        if animal_choice == '1':
            predicted_class, predicted_label = classify_image(image_path, dog_model, dog_class_labels)
            print(f"강아지 피부질환 분류 결과: 클래스 {predicted_class} ({predicted_label})")
        elif animal_choice == '2':
            predicted_class, predicted_label = classify_image(image_path, cat_model, cat_class_labels)
            print(f"고양이 피부질환 분류 결과: 클래스 {predicted_class} ({predicted_label})")

        # 3. 세부 질병 보기 및 상세 정보 제공
        view_disease = input("세부 질병 보기를 원하시나요? (y/n): ")
        if view_disease.lower() == 'y':
            selected_disease = view_disease_list(animal_choice)
            if selected_disease:
                show_disease_details(selected_disease, animal_choice)

                # 사용자에게 처음으로 돌아가기 또는 종료 여부를 물어봄
                next_action = input("처음으로 돌아가시겠습니까? (처음으로 돌아가기: '1', 종료: 'exit'): ")

                if next_action == '1':
                    continue  # 처음으로 돌아가서 다시 메인 메뉴를 출력
                elif next_action.lower() == 'exit':
                    print("프로그램을 종료합니다.")
                    return  # 프로그램 종료
            else:
                continue  # 질병 정보가 없으면 다시 메뉴로 돌아감
        else:
            continue  # 사용자가 질병 정보를 보지 않기를 원할 경우 다시 메뉴로 돌아감


if __name__ == "__main__":
    main()