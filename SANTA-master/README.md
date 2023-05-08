# SANTA

## 딥러닝 기반의 산악사고 위험 예측 시스템
- 2019 빅데이터 청년인재 (빅데이터 기반의 지능 정보시스템 개발자 과정 - 고려대학교) 최우수상 작품



### SANTA
System for Analysis of National Trekking Accidents - SANTA  
### 산타의 의미  
'산을 타다' 라는 의미의 산타, 대한민국 3000만 등산인구에게 선물을 제공한다는 의미의 산타

### 시스템 개요
- 날씨와 과거 산악사고 발생 이력 데이터 등을 활용하여 산악사고 위험도를 예측
- 등산객에게 일어날 수 있는 산악 사고의 위험도를 미리 알림
- 위험 지역에 대한 효율적인 관리 방향 제시 가능

### 데이터 출처
1) 소방청 산악사고 데이터 - 소방청
2) 종관기상관측정보 - 기상청

### 모델
- FCNN (Fully Connected Neural Network)

### 프레임워크
1) 분석 및 전처리 - Python, numpy, pandas, sklearn, xgboost
2) 모델 개발 - tensorflow, keras
3) 시스템 개발 - flask, dash, boostrap, plotly, javascript, jquery 

### 시스템

#### 메인화면 (카카오 API 이용)
![메인화면](https://user-images.githubusercontent.com/52397521/79839086-fa1ed080-83ee-11ea-8071-c004982eb976.png)

#### 텐서보드를 통한 모델 웹 시각화
![텐서보드 웹 시각화](https://user-images.githubusercontent.com/52397521/79839076-f723e000-83ee-11ea-9ce7-8ace1a7db17a.png)

#### Dash를 이용한 분석 탭
![Dash 분석 탭](https://user-images.githubusercontent.com/52397521/79839081-f8550d00-83ee-11ea-8d19-93634f57517d.png)

#### 기타정보
![기타 정보](https://user-images.githubusercontent.com/52397521/79839084-f8eda380-83ee-11ea-8f89-2b1821de78a3.png)

