# 2024 연구데이터 분석활용 경진대회
** [최우수상: 과학기술정보통신부 장관상](https://www.news1.kr/local/daejeon-chungnam/5550303)**
## 1. 개요: 딥러닝 기반 반려동물 피부질환 자가 진단 서비스
* 팀명: 둘둘즈
* 팀원: [김황민](https://github.com/NIXKim), [박세연](https://github.com/irina0627), [황성아](https://github.com/SungaHwang)
* 기간: 2024.07 - 2024.09
* 분석도구: Dataon 분석환경, Python, HTML, CSS, JS
* 소개: 반려동물의 피부 사진을 업로드하여 쉽게 자가 진단할 수 있는 서비스를 제공함으로써 반려동물의 건강을 유지하고 반려인의 편의성 증대를 위한 서비스

## 2. 주요 역할
* 김황민- 발표
* 박세연- 서비스 기획, 이상탐지 모델 개발, 웹페이지 개발, 모델 개발 메뉴얼 작성, PPT제작
* 황성아- 서비스 기획, 데이터 전처리, 피부질환 분류모델 개발, LLM을 통한 유사 질병매칭, 시스템 개발

## 3. 서비스 알고리즘

### 3-1) 시스템 구조
<img src = "https://github.com/user-attachments/assets/c0d2652d-f548-4953-9b58-22c359c8b61c" width="600"><br>
VAE를 통해 이상치 탐지를 진행하여 반려견과 반려묘가 아닌 사진을 필터링하고,
입력된 이미지 분류를 위한 다양한 딥러닝 모델(Inception-v4, ConvNeXt-v2 등)을 파인튜닝합니다.
이를 통하여 반려동물의 피부 사진을 분류하고,
분류된 카테고리와 유사한 피부질환을 Dataon의 동물질병 DB에서 LLM API를 통해 매칭함으로써 관련 정보를 제공하는 구조로 되어 있습니다.

### 3-2) 시스템 예시
<img src = "https://github.com/user-attachments/assets/01a0e659-954d-448f-9a88-5808bbcbaa0a" width="600">

## 4. 활용 데이터
### 4-1) DataON의 동물질병 DB
- 링크: https://dataon.kisti.re.kr/search/view.do
- 1689개의 반려동물(반려견, 반려묘) 질병
- 질병의 정의, 원인, 발병기전, 발병 및 역학, 주요증상, 진단, 감별진단, 병리소견, 치료, 예방, 예후 등의 정보 제공

### 4-2) AI-Hub의 반려동물 피부 질환 데이터
- 링크: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=561
- 반려동물(반려견, 반려묘) 10,000마리 이상
- 반려견 7종, 반려묘 4종 피부질환으로 판정된 반려동물 질환 이미지
- 반려동물 이미지 총 500,000장 이상
- 질환 이미지(구진, 플라크, 비듬, 각질, 태선화, 농포, 여드름, 궤양, 결절 등) 25,000장 이상

## 5. 시스템 매뉴얼
### 5-1) 디렉토리 구조
<img src = "https://github.com/user-attachments/assets/a2d2aee3-1770-4e8d-aacc-9e46a953740d" width="600">

### 5-2) 기본 설정
* 터미널 실행
* pip install –r MyFiles/requirements.txt 명령어를 통해 필요한 라이브러리 설치

### 5-2) 시스템 실행
* 시스템 실행을 통하여 프로젝트의 플로우에 맞춘 시스템을 동작할 수 있음
* python “MyFiles/system/app.py”를 통해 시스템을 실행<br>

https://github.com/user-attachments/assets/eabfe0ef-adeb-4c38-9e09-a5648843ebc6

### 5-3) 웹 페이지 실행
https://github.com/user-attachments/assets/8cbcb9fe-7c0d-449a-8e6f-5ad25ece0934



