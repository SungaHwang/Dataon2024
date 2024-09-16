import os
import pandas as pd

# disease.csv 파일 읽기
home_dir = os.path.expanduser('~')
disease_file = os.path.join(home_dir, 'workspace', 'MyFiles', 'data', 'csv_output', 'disease.csv')
df = pd.read_csv(disease_file)

# 개와 고양이에 해당하는 질병 필터링
dog_related = df[df['animal'].str.contains('개', na=False) & ~df['animal'].str.contains('고양이', na=False)]  # '개'만 포함된 질병
cat_related = df[df['animal'].str.contains('고양이', na=False) & ~df['animal'].str.contains('개', na=False)]  # '고양이'만 포함된 질병
both_related = df[df['animal'].str.contains('개', na=False) & df['animal'].str.contains('고양이', na=False)]  # '개'와 '고양이' 둘 다 포함된 질병

# 개와 고양이 관련 질병을 각각의 파일에 기록, '둘다' 관련된 것은 개와 고양이 파일 모두에 기록
dog_final = pd.concat([dog_related, both_related], ignore_index=True)  # '개'와 '둘 다' 관련된 질병
cat_final = pd.concat([cat_related, both_related], ignore_index=True)  # '고양이'와 '둘 다' 관련된 질병

# 결과 확인
print(f"개 관련 질병: {len(dog_final)}개")
print(f"고양이 관련 질병: {len(cat_final)}개")

# CSV 파일로 저장
output_dir = os.path.join(home_dir, 'workspace', 'MyFiles', 'data', 'csv_output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dog_diseases_file = os.path.join(output_dir, 'dog_diseases.csv')
cat_diseases_file = os.path.join(output_dir, 'cat_diseases.csv')

dog_final.to_csv(dog_diseases_file, index=False, encoding='utf-8')
cat_final.to_csv(cat_diseases_file, index=False, encoding='utf-8')

print("개와 고양이에 관련된 질병 파일이 생성되었습니다.")


# 개와 고양이 질병 CSV 파일 읽기
dog_df = pd.read_csv(dog_diseases_file)
cat_df = pd.read_csv(cat_diseases_file)

# 필터링 조건: 각 컬럼에 대해 "피부"가 포함된 행을 찾음
#columns_to_check = ['disease_name', 'define','condition']
columns_to_check = ['disease_name', 'define']

dog_skin_related = dog_df[
    dog_df[columns_to_check].apply(lambda row: row.str.contains('피부', na=False, case=False).any(), axis=1)
]

cat_skin_related = cat_df[
    cat_df[columns_to_check].apply(lambda row: row.str.contains('피부', na=False, case=False).any(), axis=1)
]

# 필터링된 결과 확인
print(f"개 피부 관련 질병: {len(dog_skin_related)}개")
print(f"고양이 피부 관련 질병: {len(cat_skin_related)}개")

dog_skin_related.to_csv(f'{output_dir}/dog_skin_diseases.csv', index=False, encoding='utf-8')
cat_skin_related.to_csv(f'{output_dir}/cat_skin_diseases.csv', index=False, encoding='utf-8')

print("피부 관련 질병 파일이 생성되었습니다.")

