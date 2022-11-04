# 3. Stable Decision Tree with Node-level Stabilizing
노드 단위 안정화를 이용한 안정적인 의사결정나무 알고리즘 개발 및 구현  
(데이터사이언스학과 2022 데이터마이닝 강의 조교를 진행하면서 만든 프로그램이므로 Stable Decision Tree 외 다른 자료들이 혼합돼있음)

## 0. Files and Programming Skills
+ Stable Decision Tree
  - `caseStudy_part2.ipynb` : Stable decision tree와 Stable concise rule inuction 사용 예제
  - `modules/stableDT.py` : Python으로 구현한 Stable decision tree 알고리즘 및 트리 다이어그램 시각화
+ Others for datamining TA : 
  - `breast-cancer-wisconsin.csv` : UCI의 유방암 데이터셋
  - `BC_preprocessed.csv` : breast-cancer-wisconsin 전처리 데이터셋
  - `caseStudy_part1.ipynb` : Python으로 유방암 데이터셋에 대한 EDA 및 시각화
  - `modules/splitcriterion.py` : Python으로 구현한 의사결정나무의 one-sided-maximum 분기 기준
  - `modules/stableCRI.py` : Python으로 구현한 안정적인 concise rule induction 알고리즘
  - `modules/usertree.py` : Decision tree 알고리즘
  - `modules/utils.py` : 유틸리티
  - `modules/visgraph.py` : Python 라이브러리 grapviz로 tree를 시각화하기 위한 포맷 변환
  
## 1. Introduction
- 목표 : 해석적 관점에서 안정적인 의사결정나무 알고리즘 개발
- 동기 : 의사결정나무는 dataset의 약간의 변화에도 결과가 매우 달라질 수 있는 불안정한 알고리즘이다. 이 불안정함때문에 온라인-학습에서 어떤 의사결정나무 결과라 해석해야할 지 정하기 어렵다. 이를 해결하고자 노드 단위 안정화를 이용해 보다 안정적인 의사결정나무를 생성하고자 한다.

## 2. Methods
- 노드 단위 안정화
  - 노드에서 분기 시 부트스트랩핑된 데이터셋에 대한 여러 분기 기준을 생성 
  - 생성된 분기 기준 중 최빈 분기 기준을 추출
  - 이후 계속해서 노드 단위 안정화를 재귀적 분할 반복

## 3. Result
- Stable한 Decision tree 알고리즘과 트리 다이어그램 시각화 기능

## 4. code example
- `caseStudy_part2.ipynb`의 1. Stable DT에 해당하는 부분
