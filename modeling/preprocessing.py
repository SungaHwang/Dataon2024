import os
import json
import shutil
import random
from PIL import ImageFile
from PIL import Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 사용자가 4가지 케이스 중 하나를 선택할 수 있도록
print("옵션을 선택하세요:")
print("1: TS02 - 반려견 - 피부 - 일반카메라 - 유증상 (Train)")
print("2: TS02 - 반려묘 - 피부 - 일반카메라 - 유증상 (Train)")
print("3: VL01 - 반려견 - 피부 - 일반카메라 - 유증상 (Test)")
print("4: VL01 - 반려묘 - 피부 - 일반카메라 - 유증상 (Test)")
option = input("옵션 번호를 입력하세요 (1/2/3/4): ")

# data_dir 및 train_new_dir 설정
if option == '1':
    data_dir = "/mnt/workspace/MyFiles/data/TS02/반려견/피부/일반카메라/유증상/"
    merged_dir = "/mnt/workspace/MyFiles/data/TS02/반려견/피부/일반카메라/유증상/merged"
    train_new_dir = "/mnt/workspace/MyFiles/data/train_new/dog/o"
    is_train = True
elif option == '2':
    data_dir = "/mnt/workspace/MyFiles/data/TS02/반려묘/피부/일반카메라/유증상/"
    merged_dir = "/mnt/workspace/MyFiles/data/TS02/반려묘/피부/일반카메라/유증상/merged"
    train_new_dir = "/mnt/workspace/MyFiles/data/train_new/cat/o"
    is_train = True
elif option == '3':
    data_dir = "/mnt/workspace/MyFiles/data/VL01/반려견/피부/일반카메라/유증상/"
    merged_dir = "/mnt/workspace/MyFiles/data/VL01/반려견/피부/일반카메라/유증상/merged"
    train_new_dir = "/mnt/workspace/MyFiles/data/test_new/dog/o"
    is_train = False
elif option == '4':
    data_dir = "/mnt/workspace/MyFiles/data/VL01/반려묘/피부/일반카메라/유증상/"
    merged_dir = "/mnt/workspace/MyFiles/data/VL01/반려묘/피부/일반카메라/유증상/merged"
    train_new_dir = "/mnt/workspace/MyFiles/data/test_new/cat/o"
    is_train = False
else:
    print("잘못된 옵션 번호입니다. 프로그램을 종료합니다.")
    exit()

# 경로 존재 여부 확인
if os.path.exists(data_dir):
    print("경로가 존재합니다.")
else:
    print("경로가 존재하지 않습니다.")
    exit()

# 병합된 폴더가 없으면 생성
if not os.path.exists(merged_dir):
    os.makedirs(merged_dir)

# 폴더 병합 로직
for folder_name in os.listdir(data_dir):
    if folder_name == 'merged':
        continue
        
    if '잔여' in folder_name:
        remain_folder = os.path.join(data_dir, folder_name)
        base_folder_name = folder_name.replace('_잔여', '')
        base_folder = os.path.join(data_dir, base_folder_name)

        merged_folder = os.path.join(merged_dir, base_folder_name)
        if not os.path.exists(merged_folder):
            os.makedirs(merged_folder)

        if os.path.exists(base_folder):
            print(f"폴더 병합 중: {remain_folder} + {base_folder} -> {merged_folder}")
            
            for file_name in os.listdir(base_folder):
                src_file = os.path.join(base_folder, file_name)
                dst_file = os.path.join(merged_folder, file_name)
                if os.path.isfile(src_file):
                    shutil.copy(src_file, dst_file)

            for file_name in os.listdir(remain_folder):
                src_file = os.path.join(remain_folder, file_name)
                dst_file = os.path.join(merged_folder, file_name)
                if os.path.isfile(src_file):
                    if os.path.exists(dst_file):
                        print(f"파일이 이미 존재합니다: {dst_file}. 덮어쓰지 않음.")
                    else:
                        shutil.copy(src_file, dst_file)
        
        else:
            print(f"기본 폴더가 존재하지 않습니다: {base_folder}")

    else:
        base_folder = os.path.join(data_dir, folder_name)
        merged_folder = os.path.join(merged_dir, folder_name)
        
        if not os.path.exists(merged_folder):
            os.makedirs(merged_folder)
            
        if os.path.exists(base_folder) and os.path.isdir(base_folder):
            print(f"폴더 복사 중: {base_folder} -> {merged_folder}")
            for file_name in os.listdir(base_folder):
                src_file = os.path.join(base_folder, file_name)
                dst_file = os.path.join(merged_folder, file_name)
                if os.path.isfile(src_file):
                    shutil.copy(src_file, dst_file)

# undersampling - train: 8,460 / test: 2,000 개로 통일
target_count = 8460 if is_train else 2000

# train_new 폴더가 없으면 생성
if not os.path.exists(train_new_dir):
    os.makedirs(train_new_dir)

# 카테고리별로 undersampling하여 저장
for category_folder in os.listdir(merged_dir):
    category_path = os.path.join(merged_dir, category_folder)
    
    if os.path.isdir(category_path):
        img_files = [f for f in os.listdir(category_path) if f.endswith(".jpg")]
        json_files = [f for f in os.listdir(category_path) if f.endswith(".json")]

        if len(img_files) > target_count:
            sampled_files = random.sample(img_files, target_count)
        else:
            sampled_files = img_files

        new_category_dir = os.path.join(train_new_dir, category_folder)
        if not os.path.exists(new_category_dir):
            os.makedirs(new_category_dir)

        for img_file in sampled_files:
            img_path = os.path.join(category_path, img_file)
            dst_img_path = os.path.join(new_category_dir, img_file)
            shutil.copy(img_path, dst_img_path)

            json_file = img_file.replace(".jpg", ".json")
            if json_file in json_files:
                json_path = os.path.join(category_path, json_file)
                dst_json_path = os.path.join(new_category_dir, json_file)
                shutil.copy(json_path, dst_json_path)

        print(f"카테고리 '{category_folder}'에서 {len(sampled_files)}개의 이미지와 JSON 파일을 저장했습니다.")
