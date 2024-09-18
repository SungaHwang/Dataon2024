import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from timm import create_model
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast  # Mixed Precision Training


# 경로 설정
train_dir = '/mnt/workspace/MyFiles/data/train_new/cat/o'
val_dir = '/mnt/workspace/MyFiles/data/test_new/cat/o'  # 검증 데이터 경로
batch_size = 128
num_epochs = 100 
num_classes = 3  # 분류할 클래스 수
early_stopping_patience = 3  # Early stopping patience
accumulation_steps = 4  # Gradient Accumulation 스텝 수

checkpoint_dir = '/mnt/workspace/MyFiles/checkpoints/'
checkpoint_path = os.path.join(checkpoint_dir, 'val2_cat_best_checkpoint.pth')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터 전처리
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # ConvNeXt의 기본 입력 크기
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 커스텀 데이터셋 클래스
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
                                label = label.replace(' ','').strip()
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

# 데이터셋 및 데이터로더 설정
train_dataset = CustomImageDataset(image_dir=train_dir, transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = CustomImageDataset(image_dir=val_dir, transform=data_transforms['val'])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ConvNeXt
model = create_model('convnextv2_base', pretrained=True, num_classes=num_classes)
model = model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# ReduceLROnPlateau: 성능이 개선되지 않을 때 학습률 감소
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Mixed Precision Training을 위한 GradScaler
scaler = GradScaler()

# Early Stopping 구현
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 체크포인트에서 로드하는 기능 추가
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_acc = checkpoint['accuracy']
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {epoch}, accuracy {best_acc})")
        return epoch, best_acc
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}', starting from scratch")
        return 0, 0.0  # 초기 에포크, 초기 정확도 반환

    
# 모델 학습 함수
def train_model(model, criterion, optimizer, num_epochs, patience):
    start_epoch, best_acc = load_checkpoint(model, optimizer, checkpoint_path)
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Gradient Accumulation 적용
        optimizer.zero_grad()
        accumulation_counter = 0  # 누적된 배치 수

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with autocast():  # Mixed Precision Training
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()

            # 경고 해결을 위해 모든 파라미터의 gradient를 contiguous()로 변환
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.contiguous()

            accumulation_counter += 1

            # 설정된 accumulation_steps에 도달하면 optimizer step
            if accumulation_counter % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0) * accumulation_steps
            running_corrects += torch.sum(preds == labels.data)
        
            progress_bar.set_postfix(loss=loss.item() * accumulation_steps)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        # 검증 데이터셋 평가
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        # 검증 성능이 개선될 때만 체크포인트 저장
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"Best model saved with validation accuracy: {best_acc:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc
            }, checkpoint_path)
        
        # Early Stopping 체크
        scheduler.step(val_loss)
        early_stopping(val_loss)
        
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
    return model


# 모델 학습
trained_model = train_model(model, criterion, optimizer, num_epochs, early_stopping_patience)

# 학습 완료 후 모델 저장
model_save_dir = '/mnt/workspace/MyFiles/models'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)  # 경로가 없으면 폴더를 생성합니다.

model_save_path = os.path.join(model_save_dir, 'classification_cat_convnextv2_base_model.pth')
torch.save(trained_model.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")

