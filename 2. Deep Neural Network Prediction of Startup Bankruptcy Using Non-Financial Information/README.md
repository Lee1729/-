# Deep Neural Network Prediction of Startup Bankruptcy Using Non-Financial Information
비재무정보를 이용해 신생 기업의 부도를 심층 인공신경망으로 예측  
team : 이정언, 제갈홍, 최지현 (서울과학기술대)

## 0. Files and Programming Skills
+ `Over Sampling.ipynb` : Python의 pandas, numpy를 이용해 dataset 정보 및 오버샘플링
+ `FNN.ipynb` : 딥러닝 라이브러리 pytorch를 이용한 FNN 학습 및 예측 결과
+ `Logistic Regression.ipynb` : 머신러닝 라이브러리 scikit-learn을 이용한 Logistic Regression 학습 및 예측 결과

## 1. Introduction
- 목표 : 비재무 정보를 이용해 심층 인공신경망을 통해 신생 기업의 부도를 예측
- 동기 : 신생 기업의 경우 재무 정보뿐만이 아니라 비재무정보인 대표자의 학력, 성별, 자산 등과 같은 정보가 부도 예측에 영향이 클 수 있음. 또한 부도 예측 및 신용 위험 평가는 주로 재무 정보를 활용한 전통적인 통계적 방법을 이용하는데, 비재무 정보를 이용할 때는 전통적인 통계적 방법보다 인공신경망이 더 좋은 성능을 낼 것으로 기대함.
- Dataset : 6년 이내 개인 기업에 대한 신용보증기금 신용평가 데이터

## 2. Process
- Drop out을 이용한 feed-foward network를 학습
- 학습 후 예측 성능을 logistic regression과 비교
- 비재무정보와 재무정보를 활용할 때의 비교(구현되지 않음)

## 3. Result
- FNN 
  - ACC : 0.50, Recall 0.70, AUC : 0.59
- Logistic regression
  - ACC : 0.71, Recall 0.38, AUC : 0.57
- FNN이 실제 부도 기업을 부도로 예측한 비율(Recall)이 logistic regression보다 0.32 높음

## 4. Conclusion
- FNN과 로지스틱 회귀 모두 비재무정보를 이용한 신생기업 부도 발생 예측 시 성능이 뛰어나진 않음 
- 비재무정보엔 FNN이 좀 더 좋은 성능을 보일 것으로 기대됨
- 재무 정보를 바탕으로 한 부도 예측 성능과 비교해 볼 필요가 있음 
