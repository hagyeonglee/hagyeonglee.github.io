---
emoji: 🌟
title: WomenTechStars2022 - Rising Stars
date: '2022-02-25 19:00:00'
author: gyoong
tags: MISC
categories: MISC
---

# Rising Stars 2022

> https://womentechstars.github.io/index.html 

👆 자세한 내용은 위의 공식 홈페이지에서 확인할 수 있다

### 2월 25일에 Google의 지원을 받아 국내 대학 AI/CS/EE 분야 여학생을 대상으로 개최되는 여성 과학자 동계 학술워크샵을 참석했다! 

대학원생분들이 현재 진행하고 계시는 연구를 발표하시면 그에 대해서 나를 비롯한 청중과 패널 교수님들께서 궁금한 부분에 대해서 질문과 피드백을 하신다. 연구외적으로도 논문, 발표 등에 대해서 박사과정의 대학원생분들 뿐만아니라 교수님들께 팁이나 경험담을 들을 수 있는 좋은 시간이었다. 혹시 위 행사를 들으려고 고민하시는 분은 꼭 한 번 들어보시기를 추천한다 👍

### Rising stars 논문
논문 발표 시간은 이미지 / 영상 처리와 머신러닝 / 센서 / 보안 세션으로 나뉜다. 1시간 동안 본인이 관심있는 세션 방에 들어가서 논문 발표를 듣고 질의응답을 자유롭게 할 수 있다. 나는 머신러닝 / 센서 / 보안 세션에 들어가서 논문 발표를 들었다. 최근 관심을 갖게된 분야인 AutoML과 관련된 NAS(Neural Architecture Search) 연구 내용이 궁금했기 때문이다. 

> 아래 내용들을 간단하게 note taking한 내용들이다.

###  1. "Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets"

- NAS : Optimal 한 모델 아키텍처 를 AI가 neural architecture 를 탐색하려고 하는 분야
- NAS로 찾은 모델이 더 효율적 
- Search space : 최적의 모델을 찾기위해서 디자인하는 것, 각 stage 별로 layer를 몇개 정할지, kernel의 사이즈를 얼만큼 정할지, 모든 경우의 수를 따지면 10의 19제곱정도 됨
- GPU를 많이 사용해서 train함 -> 현실적으로 어려움 -> 어떻게 하면 효율적으로 train할 수 있을까를 고민
- task specific
- Target dataset이 바뀌면 다시 처음부터 neural architecture를 찾기 위해 처음부터 학습을 해야하는 점이 문제점 -> 이를 해결할 수 있는 방법으로 meta-learning 적용
- 👀 
    
    Meta-learning으로 NAS 모델을 학습시켜서 일반화시켜서 추가학습 없이 걸맞는 뉴럴 네트워크를 찾아주는 방법을 논문으로 씀

    - 드라이빙 스킬을 한 번 배우면 새로운 자동차에도 빠르게 적용가능
    - 여러 task에 대해 일반화할 수 있는 방법 
    - 여러 dataset과 그에 맞는 architecture 를 준비 

- 주요 키워드
    - Set-Encoding : dataset을 일정한 하나의 벡터로 표현
    - Graph decoder
    - Neural network transformation

### 2. Jointly Processing Image and Video Restoration Tasks Using Deep Learning

- edge나 text는 고주파 디테일 데이터임
- SDR -> HDR 은 색상을 확장해야함
- Pixel 위치마다 local한 contrast를 함께
- serial과 joint 하는 방법이 있음
    - 👀  
    task가 관련이 있고 함께 학습하면 개선이 더 빠르다는 것을 joint하여 논문으로 작성
- ResBlock with Modulation

### 3. Job Talk & Career
그리고 이후 연달아서는 KAIST의 김주호 교수님께서 Job Talk를 잘하기 위한 팁에 대해서 발표를 해주셨고 USF의 권창현 교수님께서 좋은 논문을 작성하는 방법에 대해서 발표를 해주셨다.

> 김주호 교수님 발표 note 

- 첫인상을 좌우하는 talk
- 학교에서 평가
- 예측 task
- Set-up : 내가 하는 분야가 왜 중요한지
- 중/단기 plan을 담는 것
- Tell a story
- 다시 주제로 come-back할 수 있는 공간
- 듣는 사람이 주제 파악이 쉽도록 map을 갖는 게 필요 : keyword가 있으면 좋음 overview 역할
- 마지막 슬라이드에 thank you 쓰지말기
    - 지루함 -> 화면에 핵심 포인트, summary를 넣으면 좋음
    - story가 흘러가고 slide가 보조
    - Script 읽지 말기 
    - 커넥션을 만들어가는 것
    - 완전히 다른 연구실에 가서 practice talk을 하는 것
    - 녹화하면서 고쳐나가기

> 권창현 교수님 발표 note 

- 좋은 연구를 하는 방법
    - 논문에 내가 무슨 말을 하고 싶은지 알아야한다
    - 연구가 끝나지 않았더라도 논문을 한 번 써보는 것
    - 연구 주제에 대해 3단계로 말해보기
        - 당신은 무엇에 대해 공부하고 있나요?
        - 왜 그 주제를 공부하고 있지요?
        - So what? - 그걸 알면 뭐가 어쨌다는 거죠?
            - 연구가 성공하면 일어나는 일들
    - 논문쓰는 것은 결국 연구의 iteration의 일부라는 것!
    - 내 스토리를 이야기 하는 것이 좋다
        - 어떤 부분에 서포트가 필요하면 가져다가 쓰는 것
        - Latex ; parenthetical citation
    - 원병묵 교수님의 과학 논문 쓰는 법

