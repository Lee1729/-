# Causal analysis of online community interest and stock price volatility
온라인 커뮤니티의 특정 주식 종목에 대한 관심도와 주가의 변동성, 거래량 간 인과관계를 분석

## 0. File and Programming Skills
- ``
- ``
- ``

## 1. Introduction 
- Goal : 
- 동기 : 커뮤니티의 글을 주식 거래에 참고할 수 있는 데이터로 활용할 수 있지 않을까?
- dataset : 네이버 금융 종목토론실의 게시글, 야후 API finance API의 주식 데이터 

## 2. Process
- 시가총액을 기준으로 3개의 주식 종목(삼성전자, 셀트리온, 신풍제약) 선정
- 웹크롤링을 통해 네이버 종목토론실의 해당종목 게시판에서 3년(2017.7~2020.11)동안 작성된 글의 날짜 데이터를 수집
- 야후 finance API의 해당 주식 종목 데이터(날짜,주가,거래량)를 수집
- 해당 주식 종목의 변동성과 증감률을 계산(변동성 = 한 주동안의 (최고가-최저가) / 최저가)
- 커뮤니티 글 수와 변동성, 거래량 상관분석, 인과관계 분석 및 시각화

## 3. Analysis
- 
