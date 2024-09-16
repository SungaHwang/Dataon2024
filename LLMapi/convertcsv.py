import subprocess
import os

# CSV 파일을 저장할 디렉토리 설정
home_dir = os.path.expanduser('~')
output_dir = os.path.join(home_dir, 'workspace', 'MyFiles', 'data', 'csv_output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# 테이블 목록 가져오기
result = subprocess.run(['mdb-tables', 'MyFiles/data/05동물질병.accdb'], stdout=subprocess.PIPE)
tables = result.stdout.decode('utf-8').split()

# 각 테이블을 CSV로 변환
for table in tables:
    if table.strip():  # 테이블 이름이 빈 문자열이 아닐 때만 실행
        csv_file = os.path.join(output_dir, f"{table}.csv")
        try:
            with open(csv_file, 'w') as f:
                subprocess.run(['mdb-export', 'MyFiles/05동물질병.accdb', table], stdout=f)
            print(f"Table {table} saved as {csv_file}")
        except PermissionError:
            print(f"Permission denied: Could not write to {csv_file}")
        except Exception as e:
            print(f"An error occurred while processing table {table}: {e}")
    else:
        print("Empty table name encountered, skipping...")

print("All tables processed.")
