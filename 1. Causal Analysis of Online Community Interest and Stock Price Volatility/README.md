# Causal Analysis of Online Community Interest and Stock Price Volatility
온라인 커뮤니티의 특정 주식 종목에 대한 관심도와 주가의 변동성, 거래량 간 인과관계를 분석

## 0. File and Programming Skills
- `crawling_naver.ipynb` : BeautifulSoup를 이용한 웹 크롤링으로 네이버 금융의 종목게시판 글의 날짜 데이터 수집 
- `convert_data.ipynb` : 글 날짜 데이터셋을 일주일별 글 수로 취합
- `data_analysis.ipynb` : python의 pandas, statmodels를 이용해 상관관계 및 인과관계 분석과 seaborn을 이용한 시각화

## 1. Introduction 
- Goal : 온라인 커뮤니티 글 수와 가격 변동성, 거래량의 관계를 분석
- 동기 : 커뮤니티의 글을 주식 거래에 참고할 수 있는 데이터로 활용할 수 있지 않을까?
- dataset : 네이버 금융 종목토론실의 게시글, 야후 API finance API의 주식 데이터 

## 2. Process
- 시가총액을 기준으로 3개의 주식 종목(삼성전자, 셀트리온, 신풍제약) 선정
- 웹크롤링을 통해 네이버 종목토론실의 해당종목 게시판에서 3년(2017.7~2020.11)동안 작성된 글의 날짜 데이터를 수집
- 야후 finance API의 해당 주식 종목 데이터(날짜,주가,거래량)를 수집
- 해당 주식 종목의 변동성과 증감률을 계산(변동성 = 한 주동안의 (최고가-최저가) / 최저가)
- 커뮤니티 글 수, 거래량, 가격 변동성의 상관분석, 인과관계 분석 및 시각화

## 3. Analysis
+ 글 수는 거래량보다 가격 변동성에 대해 유의한 정보일 수 있을까?
+ 상관분석(삼성전자)
  - (글 수, 가격 변동성)은 0.61 (거래량, 가격 변동성)은 0.73으로 거래량이 글 수보다 가격변동성과 더 상관관계가 강한 편
+ 인과분석(삼성전자)
  - 글 수가 거래량보다 가격 변동성에 대해 인과적일 수 있을까?
  - 글 수 -> 가격 변동성 : 공적분 p-value = 0.006, Granger Causality p-value = 0.314
  - 가격 변동성 -> 글 수 : 공적분 p-value = 0.000, Granger Causality p-value = 0.000
  - 거래량 -> 가격 변동성 : 공적분 p-value = 0.834, Granger Causality p-value = 0.139
  - 가격 변동성 -> 거래량 : 공적분 p-value = 0.056, Granger Causality p-value = 0.477
  - 공적분 p-vlaue를 비교해볼 때, 글 수는 가격변동성에 대해 거래량보다 장기적인 관계성이 있을 수 있음
  - 유의수준 0.95에서 가격변동성 -> 글 수 Granger Causality가 성립함
  
## 4. Conclusion
- 글 수는 가격 변동성 및 거래량과 양의 상관관계를 띔
- 그랜저 인과성 검정 결과로 볼 때 가격 변동성은 글 수에 대한 인과적 요인이 될 수 있음
- 특정 종목에서는 글 수가 거래량보다 가격 변동성에 대해 더 유의한 정보를 줄 수도 있음
- 결론적으로 주식 커뮤니티의 글 수는 실제로 주식 시장참여자의 거래행위와 관련이 있는 것으로 보임 
- 3개 종목이 아닌 더 많은 case study가 필요
