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

# 1. 평가를 위한 데이터 전처리
#test_dir = '/mnt/workspace/MyFiles/data/test_new/dog/o'  # 테스트 데이터 경로
test_dir = '/mnt/workspace/MyFiles/data/test_new/cat/o'  # 테스트 데이터 경로
batch_size = 128
num_classes = 3  # 분류할 클래스 수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((299, 299)),  # Inception 모델의 입력 크기
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

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

test_dataset = CustomImageDataset(image_dir=test_dir, transform=data_transforms['test'])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. Inception-v4 모델 불러오기
model = create_model('inception_v4', pretrained=False, num_classes=num_classes)
checkpoint = torch.load('MyFiles/models/classification_cat_inception_v4_model.pth')
#checkpoint = torch.load('MyFiles/checkpoints/cat_best_checkpoint.pth')
model.load_state_dict(checkpoint) 
#model.load_state_dict(checkpoint['model_state_dict'])  # 가중치만 로드
model = model.to(device)
#model.load_state_dict(torch.load('MyFiles/checkpoints/dog_best_checkpoint.pth'))  # 학습된 모델 로드
# model.eval()

# 모델을 TorchScript로 변환하여 최적화
scripted_model = torch.jit.script(model)
scripted_model.eval()

# 3. 평가 메트릭 계산
all_labels = []
all_preds = []

with torch.no_grad():
    # tqdm을 사용하여 프로그레스 바 추가
    for inputs, labels in tqdm(test_loader, desc="Processing", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = scripted_model(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# 4. 정확도, 재현율, F1 스코어, 민감도 계산
accuracy = accuracy_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds, average='macro')  # 재현율
f1 = f1_score(all_labels, all_preds, average='macro')  # F1 스코어
precision = precision_score(all_labels, all_preds, average='macro')  # 민감도

# 결과 출력
output_file = "MyFiles/results/classification_cat_inceptionV4_evaluation.txt"
#output_file = "MyFiles/results/cat_inceptionV4_evaluation.txt"
with open(output_file, "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Recall (재현율): {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Precision (민감도): {precision}\n")

print(f"Results saved to {output_file}")
