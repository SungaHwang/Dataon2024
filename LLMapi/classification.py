import os
import re
import time
import requests
import pandas as pd

# HuggingFace API 설정
HUGGINGFACE_API_KEY = ""
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
}

dog_class_labels = {0: "비듬_각질_상피성잔고리 (Dandruff_Keratin_Epithelial Colloid)", 1: "태선화_과다색소침착 (Lichenification_Hyperpigmentation)", 2: "농포_여드름 (Pustules_Acne)", 3: "미란_궤양 (Erosion_Ulcer)", 4: "결절_종괴 (Nodule_Tumor)"}

cat_class_labels = {0: "비듬_각질_상피성잔고리 (Dandruff_Keratin_Epithelial Colloid)", 1: "농포_여드름 (Pustules_Acne)", 2: "결절_종괴 (Nodule_Tumor)"}

def query_huggingface_api(text, class_labels):
    labels = list(class_labels.items())
    labels_str = ", ".join([f"{v}: {k}" for k, v in labels])
    prompt = f"""
    Disease description: '{text}'
    Please choose the label number of the disease class that is most similar:
    {labels_str}
    Make sure to return only one label number.
    """
    
    payload = {
        "inputs": prompt
    }
    
    for i in range(5):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                # 응답 성공, 루프 중단
                break
            elif response.status_code == 503 and 'Model' in response.json().get('error', ''):
                # 모델이 로딩 중일 경우
                estimated_time = response.json().get('estimated_time', 300)
                print(f"모델이 로딩 중입니다. {i+1}번째 재시도, 예상 대기 시간: {estimated_time}초...")
                time.sleep(estimated_time)  # 모델 로딩 대기
            else:
                print(f"예기치 않은 상태 코드: {response.status_code}. 다시 시도 중...")
                time.sleep(5)  # 기타 오류 시 5초 대기 후 재시도

        except requests.exceptions.ConnectionError:
            print(f"연결 오류 발생. {i+1}번째 재시도...")
            time.sleep(5)  # 연결 오류 시 5초 대기 후 재시도

    # API 응답 확인
    try:
        response_data = response.json()
        print(f"API 응답: {response_data}")

        if isinstance(response_data, list) and len(response_data) > 0:
            generated_text = response_data[0].get('generated_text', "Unknown")
        elif isinstance(response_data, dict) and 'generated_text' in response_data:
            generated_text = response_data['generated_text']
        else:
            print("API 응답에 예상된 데이터가 없습니다.")
            return "Unknown"
        
        # 레이블 번호만 추출
        label_number = extract_label_number(generated_text, class_labels)
        return label_number
    
    except Exception as e:
        print(f"API 응답 처리 중 오류 발생: {e}")
        return "Error"

def extract_label_number(text, class_labels):
    # 모든 숫자를 추출
    numbers = re.findall(r'\b\d+\b', text)
    
    # 유효한 레이블 번호만 필터링
    valid_numbers = [num for num in numbers if int(num) in class_labels]
    
    # 가장 뒤에 언급된 유효한 레이블 번호 반환
    if valid_numbers:
        return valid_numbers[-1]
    return "Unknown"

def map_label_to_category(label_number, class_labels):
    try:
        return class_labels.get(int(label_number), "Unknown")
    except ValueError:
        return "Unknown"

def classify_disease(df, class_labels, animal_type):
    classified_diseases = []

    for idx, row in df.iterrows():
        disease_name = row['disease_name']
        print(f"\n처리 중인 {animal_type} 질병: {disease_name}")

        # LLM API를 통해 클래스 레이블을 예측
        label_number = query_huggingface_api(disease_name, class_labels)
        print(f"'{disease_name}'에 대한 API 응답 레이블 번호: {label_number}")

        # 레이블 번호를 카테고리로 변환
        assigned_label = map_label_to_category(label_number, class_labels)
        
        if assigned_label != "Unknown":
            classified_diseases.append({
                'disease_name': disease_name,
                'assigned_label': assigned_label
            })
            print(f"'{disease_name}'는 '{assigned_label}'로 분류되었습니다.")
        else:
            print(f"'{disease_name}'는 관련성이 낮아 분류되지 않았습니다.")

        # 중간 결과 출력
        print(f"현재까지 {animal_type} 분류된 질병 수: {len(classified_diseases)}")
        print(f"현재 분류된 질병 목록:")
        for disease in classified_diseases:
            print(f" - {disease['disease_name']}: {disease['assigned_label']}")

    return classified_diseases

# CSV 파일 경로 설정 (각각 강아지와 고양이 파일)
home_dir = os.path.expanduser('~')

# CSV 파일 경로 설정 (각각 강아지와 고양이 파일)
dog_skin_diseases_file = os.path.join(home_dir, 'workspace', 'MyFiles', 'csv_output', 'dog_skin_diseases.csv')
cat_skin_diseases_file = os.path.join(home_dir, 'workspace', 'MyFiles', 'csv_output', 'cat_skin_diseases.csv')

# 강아지와 고양이 질병 CSV 파일 읽기
dog_df = pd.read_csv(dog_skin_diseases_file)
cat_df = pd.read_csv(cat_skin_diseases_file)

output_dir = os.path.join(home_dir, 'workspace', 'MyFiles', 'results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# 강아지 질병 분류
dog_classified = classify_disease(dog_df, dog_class_labels, "강아지")
dog_classified_output = os.path.join(output_dir, 'Mistral-7B_dog_classified_diseases.csv')
pd.DataFrame(dog_classified).to_csv(dog_classified_output, index=False, encoding='utf-8')
print("강아지 질병 분류 결과가 CSV 파일로 저장되었습니다.")

# 고양이 질병 분류
cat_classified = classify_disease(cat_df, cat_class_labels, "고양이")
cat_classified_output = os.path.join(output_dir, 'Mistral-7B_cat_classified_diseases.csv')
pd.DataFrame(cat_classified).to_csv(cat_classified_output, index=False, encoding='utf-8')
print("고양이 질병 분류 결과가 CSV 파일로 저장되었습니다.")
