# 데이터 파싱 (patent_parser)

### recreateXml
##### out파일을 xml 포멧으로 변환
###### * out파일은 xml포멧 형식에 맞지 않아 파싱불가.

### XmlParser
##### xml을 파싱하여 필요한 데이터 형식으로 변환
###### * CLREF = 레퍼런스 claim정의
###### * PDAT1 = 레퍼런스 데이터의 앞부분 텍스트
###### * PDAT2 = 레퍼런스 데이터의 뒷부분 텍스트

### XmlParserForW2V
##### xml을 파싱하여 필요한 데이터 형식으로 변환 BERT MODEL 적용

### data_set 추가!!
###### * 현 디렉토리에 data_set 폴더 생성 및 파일 추가 필요



# ML APP (독립항, 종속항 구분 애플리케이션)

### 데이터 Training APP
##### BERT모델을 활용하여 청구항 구분에 맞는 FINE TUNING TRANING 수행
###### text_classification_bert/BERT.ipynb

### 데이터 Testing APP
##### TRAINING된 BERT모델을 TESTING 수행
###### text_classification_bert/BERT_eval.ipynb

### 응용 API화 된 ML APP
##### 개발된 청구항구분 ML APP을 API형태로 구성
###### text_classification_bert/server.ipynb



# RANDOM-FOREST APP (rf_project)

### RANDOM FOREST
##### rf_project/randomforest.ipynb 주피터 노트북 실행
###### 특허의 성과지표 별 단어의 영향도 분석