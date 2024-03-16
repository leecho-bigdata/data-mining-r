# 04_2.SVM(분류)_기초

library(tidyverse)
library(tidymodels)
library(skimr)           # 데이터 요약(EDA)
library(vip)             # 중요한 변수 찾기
library(GGally)          # 여러 변수 통합 그래프
library(themis)          # 불균형 자료 처리






# 2.데이터 불러오기
breast1_tb <- read_csv('유방암진단_기초.csv', 
                       col_names = TRUE,
                       locale=locale('ko', encoding='euc-kr'), # 한글
                       na=".") %>% # csv 데이터 읽어오기
  mutate_if(is.character, as.factor)

str(breast1_tb)
breast1_tb

breast1_tb <- breast1_tb %>%
  mutate(진단 = factor(진단, 
                     levels = c(1, 0),              #관심변수=Yes           
                     labels = c("악성", "양성")))






# 6.Model 만들기(모델 설정)

# 모델 인자(argument) 확인
# install.packages("LiblineaR")
args(svm_rbf) 

svm_model <- svm_linear() %>%
  set_engine("LiblineaR") %>%
  set_mode("classification")

svm_model

svm_fit <- svm_model %>%
  fit(formula = 진단 ~ 표준크기+표준질감 , 
      data = breast1_tb)


# 결과확인
svm_fit %>% 
  extract_fit_engine()


# 3.그래프 그리기
beta0 = -0.05349934
beta1 = -1.054678
beta2 = -0.06800296

breast1_tb %>%
  ggplot(mapping = aes(x = 표준크기 , 
                       y = 표준질감 , 
                       color = 진단)) +
  geom_point() +
  xlim(-3,3) +
  ylim(-3,3) +  
  geom_abline(intercept = beta0 / beta1,
              slope = -beta1 / beta2,
              color = "green") +
  geom_abline(intercept = (beta0 - 1) / beta1,
              slope = -beta1 / beta2,
              color = "blue",
              linetype = "dashed") +
  geom_abline(intercept = (beta0 + 1) / beta1,
              slope = -beta1 / beta2,
              color = "red",
              linetype = "dashed")


