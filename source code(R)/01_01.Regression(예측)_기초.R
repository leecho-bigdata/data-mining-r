# 01_00.회귀분석(예측)_기초






# 1.package 설치
# install.packages("tidyverse") 
# install.packages("tidymodels")
# install.packages("skimr")
# install.packages("vip")
# install.packages("GGally")
library(tidyverse)
library(tidymodels)
library(skimr)           # 데이터 요약(EDA)
library(vip)             # 중요한 변수 찾기
library(GGally)          # 여러 변수 통합 그래프






# 2.데이터 불러오기
housing_tb <- read_csv('주택가격_기초.csv', 
                       col_names = TRUE,
                       locale=locale('ko', encoding='euc-kr'), # 한글
                       na=".") %>% # csv 데이터 읽어오기
  mutate_if(is.character, as.factor)

str(housing_tb)
housing_tb



housing_tb %>%
  ggplot(mapping = aes(x = 면적,
                       y = 가격)) +
  geom_point(alpha = 0.5,
             color = "red") +
  xlim(5,14) +
  ylim(80,200)





# 모델 세팅
lr_model <- linear_reg() %>% 
  set_engine("lm") %>%            
  set_mode("regression")


# 9.Model 훈련(모델 학습)

# 모델 훈련
lr_fit <- lr_model %>%
  fit(formula = 가격 ~ ., 
      data = housing_tb)

# 결과확인
lr_fit %>% 
  extract_fit_engine() %>%
  summary()

# 결과확인
lr_fit %>% 
  tidy()

# 중요변수 확인
lrr_fit %>%
  extract_fit_engine() %>%
  vip()






# 10.모델 test 및 평가

# 모델 테스트
lrr_fit %>%
  predict(test_data)

# 데이터 예측
lr_fit %>%
  augment(housing_tb) %>%
  select(가격, .pred, .resid)

# 실제값와 예측값 그래프
lr_fit %>%
  augment(housing_tb) %>% 
  ggplot(aes(x = 가격, 
             y = .pred)) + 
  geom_abline(color = "red", 
              lty = 1) + 
  geom_point(alpha = 0.5) + 
  coord_obs_pred() + 
  labs(x = "observed", 
       y = "predicted")


housing_tb %>%
  ggplot(mapping = aes(x = 면적,
                       y = 가격)) +
  geom_point(alpha = 0.5,
             color = "red") +
  xlim(5,14) +
  ylim(80,200) +  
  geom_abline(intercept = 15.09,
              slope = 14.072) +
  geom_abline(intercept = 10,
              slope = 15,
              color = "green",
              linetype = 3, 
              size = 1)














