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
##### text_classification_bert/BERT.ipynb

### 데이터 Testing APP
##### text_classification_bert/BERT_eval.ipynb

### classification NOTEBOOK APP
##### server.ipynb



# RANDOM-FOREST APP (rf_project)

### RANDOM FOREST
##### 특허의 성과지표 별 단어의 영향도 분석
###### rf_project/randomforest.ipynb 주피터 노트북 실행