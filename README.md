# Capstone_Design

1. 둘다 라벨 일치시키고 드랍 하지 않고 비교
2. 둘다 라벨 일치시키고 드랍후 비교
3. 5년치만 사용 -> case 분류는 날씨/날씨x (1/0)
모델: dnn,Lstm,rnn,svm Layer param 조절

전체: 14486

실족추락: 3407 (1) 탈진·탈수: 863 (1) 일반조난: 3365 (1) 저체온증: 61 (1)

낙석·낙빙: 63 (0) 기타산악: 5255 (0) 산악기타: 14 (0) 개인(급.만성)질환: 0 (0) 개인질환: 1425 (0) 고온환경질환: 18 (0) 자살기도(산악): 0 (0) 암벽등반: 0 (0) 야생식물 섭취 중독: 15 (0)

제가 푸쉬 잘못해서 머지할 때 main이랑 이상한게 섞였어요,,그냥 코드는 우서 0501으로 확인하시고, LSTM 정확도는 model.evaluate 해서 맨 마지막에 accuracy로 뽑아놓는 부분 확인하심 돼요!
그리고 지금 문제가 h2o automl에서 aml.train(x=list(X2_train_df.columns), y=0, training_frame=h2o_train, leaderboard_frame=h2o_valid) 의 train 부분이 있는데 
y= integer || string type이 받아져야하는데 y가 1차원 dataframe 형태로 받아져요! 그래서 임의로 0을 넣었는데 이 이슈 또한 확인해봐야할 것 같아요!
