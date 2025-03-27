# construct

# OS 및 라이브러리 버전

OS : Ubuntu 22.04.5 LTS

라이브러리은 /environment 참고

## 파일 구조
/analysis : 데이터 분석 실험실
/data : 데이터가 들어가는 폴더
/environment : conda 환경 yaml 과 pip requirement 파일
/experiment : 모델이 돌아가는 폴더
/experiment/configs : 모델을 config 파일이 들어있는 폴더
/experiment/dpo_model : dpo 학습한 model parameter 가 저장되는 곳
/experiment/outputs : ckpt 저장 폴더
/experiment/result : csv 저장 폴더
/experiment/src : 소스코드
/experiment/main.py : 실행 파일 ( -m 으로 각 task 수행 가능 train, test, SCD, DPO, norm)
/experiment/src/dataset : dataset module 관련 폴더
/experiment/src/model : model 관련 폴더
/experiment/src/parameter : parameter 저장 장소
/experiment/src/rag : rag, PDF reader 관련 폴더
/experiment/src/utils : metric, logger 등 유틸 관련 폴더
/experiment/src/constraint_test.py : norm 모드의 run 파일 --> 평균 백터에서 mutate 로 답변 라벨링 코드
/experiment/src/DPO.py : DPO run 파일
/experiment/src/SCD_test.py : Self consistency decoding test run 파일
/experiment/src/test.py : 일반 추론 run 파일
/experiment/src/train.py : 학습 run 파일