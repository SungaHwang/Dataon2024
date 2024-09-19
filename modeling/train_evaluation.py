import torch
from torch.utils.data import Dataset, DataLoader 
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from torchvision import transforms
from timm import create_model
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import os
from tqdm import tqdm

# 사용자에게 옵션을 선택하게 함
print("옵션을 선택하세요:")
print("1. 강아지 - Inceptionv4")
print("2. 고양이 - Inceptionv4")
print("3. 강아지 - ConvNeXtV2 Base")
print("4. 고양이 - ConvNeXtV2 Base")

option = input("옵션 번호를 입력하세요 (1-4): ")

# 사용자 입력에 따른 데이터 경로 및 모델 설정
if option == '1':
    animal_type = 'dog'
    model_type = 'inceptionv4'
elif option == '2':
    animal_type = 'cat'
    model_type = 'inceptionv4'
elif option == '3':
    animal_type = 'dog'
    model_type = 'convnextv2_base'
elif option == '4':
    animal_type = 'cat'
    model_type = 'convnextv2_base'
else:
    raise ValueError("잘못된 입력입니다. 1-4 사이의 값을 입력하세요.")

# 평가를 위한 데이터 설정
test_dir = f'/mnt/workspace/MyFiles/data/test_new/{animal_type}/o'
batch_size = 128
num_classes = 5 if animal_type == 'dog' else 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터 전처리 설정 (모델에 맞는 입력 크기 적용)
if model_type == 'inceptionv4':
    data_transforms = transforms.Compose([
        transforms.Resize((299, 299)),  # Inceptionv4 모델의 입력 크기
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
elif model_type == 'convnextv2_base':
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # ConvNeXtV2 Base 모델의 입력 크기
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# 데이터셋 클래스 정의
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        label_set = set()

        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.jpg'):
                    img_path = os.path.join(root, file)
                    json_path = img_path.replace('.jpg', '.json')
                    if os.path.exists(json_path):
                        self.image_paths.append(img_path)
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            label_info = data.get('labelingInfo', [])
                            if label_info:
                                label = label_info[0].get('polygon', {}).get('label') or label_info[0].get('box', {}).get('label')
                                label = label.strip()
                                self.labels.append(label)
                                label_set.add(label)

        self.class_to_idx = {label: idx for idx, label in enumerate(sorted(label_set))}
        self.labels = [self.class_to_idx[label] for label in self.labels]
        
        print(f"Number of unique classes: {len(self.class_to_idx)}")
        print(f"Class to index mapping: {self.class_to_idx}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)

# 데이터 로더 설정
test_dataset = CustomImageDataset(image_dir=test_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 설정
if model_type == 'inceptionv4':
    model = create_model('inception_v4', pretrained=False, num_classes=num_classes)
    checkpoint_path = f'MyFiles/models/classification_{animal_type}_inception_v4_model.pth'
elif model_type == 'convnextv2_base':
    model = create_model('convnextv2_base', pretrained=False, num_classes=num_classes)
    checkpoint_path = f'MyFiles/models/classification_{animal_type}_convnextv2_base_model.pth'

# 모델 가중치 불러오기
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model = model.to(device)

# 모델을 TorchScript로 변환하여 최적화
scripted_model = torch.jit.script(model)
scripted_model.eval()

# 평가 메트릭 계산
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Processing", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = scripted_model(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# 정확도, 재현율, F1 스코어, 민감도 계산
accuracy = accuracy_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
precision = precision_score(all_labels, all_preds, average='macro')

# 결과 출력
output_file = f"MyFiles/results/classification_{animal_type}_{model_type}_evaluation.txt"
with open(output_file, "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Precision: {precision}\n")

print(f"Results saved to {output_file}")
