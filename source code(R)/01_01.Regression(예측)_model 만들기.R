# 01_01.회귀분석(예측)_model 만들기
# https://www.tidymodels.org/start/models/
# https://www.tmwr.org/

#######################################################
# tidyverse: tidy형식 기본도구                        #
#            ggplot2, purrr, tibble  3.0.3,           #
#            dplyr, tidyr, stringr, readr, forcats    #
#            등 8개 패키지를 한 곳에 묶은 형태        #
#                                                     #
# tidymodels:tidy 모델 분석도구(데이터마이닝, 통계)   #
#            broom, recipes, dials, rsample, infer,   #
#            tune, modeldata, workflows, parsnip,     #
#            yardstick                                #
#                                                     #
# tidytext: 텍스트마이닝 도구                         #
#           rlang, tibble, dplyr, stringr, hunspell,  #
#           generics,lifecycle, Matrix, tokenizers,   #
#           janeaustenr, purrr                        # 
#######################################################






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
housing_tb <- read_csv('주택가격.csv', 
                       col_names = TRUE,
                       locale=locale('ko', encoding='euc-kr'), # 한글
                       na=".") %>% # csv 데이터 읽어오기
  mutate_if(is.character, as.factor)

str(housing_tb)
housing_tb






# 3.data 전처리

# 필요없는 변수제거
# recipe에서 제거할 수도 있음
housing_tb <- housing_tb %>%
  select(-c(id))  

str(housing_tb)
housing_tb

# 범주형 변수(factor) 변환
housing_tb <- housing_tb %>%
  mutate(주거유형 = factor(주거유형, 
                       levels = c(1:5),                        
                       labels = c("단독주택", "2가구변경", "듀플렉스",
                                  "타운젠트바깥쪽","타운젠트안쪽")),
         판매유형 = factor(판매유형, 
                       levels = c(1,2),                        
                       labels = c("보증증서", "법원관리증서")),
         판매조건 = factor(판매조건, 
                       levels = c(1,2),                        
                       labels = c("정상판매", "압류및공매도")))

str(housing_tb)
housing_tb






# 4.데이터 탐색(EDA)






# 5.훈련용, 테스트용 데이터 분할: partition

# 데이터 partition
set.seed(123) # 시드 고정 

# 기본 75%는 훈련용, 25%는 테스트용으로 구분
# 결과변수 비율 반영
housing_split <- housing_tb %>%
  initial_split(strata = 가격) # 결과변수 비율반영

housing_split

# training, test용 분리
train_data <- housing_split %>%
  training()

test_data  <- housing_split %>%
  testing()

train_data
test_data






# 6. recipes 만들기






# 7.Model 만들기(모델 설정)

# 모델 인자(argument) 확인
args(linear_reg) 

# 모델 세팅
lrr_model <- linear_reg() %>% 
  set_engine("lm") %>%            
  set_mode("regression")

lrr_model






# 8.workflow 만들기






# 9.Model 훈련(모델 학습)

# 모델 훈련
lrr_fit <- lrr_model %>%
  fit(formula = 가격 ~ ., 
      data = train_data)

# 결과확인
lrr_fit %>% 
  extract_fit_engine() %>%
  summary()

# 결과확인
lrr_fit %>% 
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
lrr_fit %>%
  augment(test_data) %>%
  select(가격, .pred, .resid)

# 실제값과 예측값 그래프
lrr_fit %>%
  augment(test_data) %>% 
  ggplot(aes(x = 가격, 
             y = .pred)) + 
  geom_abline(color = "gray50", 
              lty = 2) + 
  geom_point(alpha = 0.5) + 
  coord_obs_pred() + 
  labs(x = "observed", 
       y = "predicted")




















