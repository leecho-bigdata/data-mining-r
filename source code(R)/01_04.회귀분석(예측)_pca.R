# 01_04.회귀분석(예측)_pca

### 추가 ###
# 7.Model 만들기(모델 설정) 수정
# 9.Model 훈련(모델 학습) 수정
# 10.모델 test 및 평가 수정





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

# 데이터 탐색: 범주형, 연속형 구분
# skimr::skim() - package명을 앞에 써서 구분
# 패키지를 여러개 사용할 경우에 이름이 같은 경우도 있어서
# 구분이 필요할 경우에 [패키지명::]을 사용
housing_tb %>%
  skimr::skim() 

# base accuracy
housing_tb %>% 
  summarise(mean = mean(가격, na.rm = T))   # 수치형 변수


# 수치형 데이터 그래프
housing_tb %>%
  select(가격:면적_2층) %>%
  ggpairs()

# 범주형 데이터 그래프
housing_tb %>%
  select(가격, 주거유형:판매조건) %>%
  ggpairs()






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

train_data %>% 
  summarise(mean = mean(가격, na.rm = T))

test_data %>% 
  summarise(mean = mean(가격, na.rm = T))






# 6.recipes 만들기
# step_normalize(all_numeric()) : 데이터 정규화
# 특히, penalty 사용하는 모델에서 중요(logistic, SVM 등)
# step_zv(all_predictors()) : 단일 고유 값 (예 : 모두 0) 변수 제거. 
# step_dummy(all_nominal(), -all_outcomes()) : one-hot-ecoding
# step_log(Gr_Liv_Area, base = 10) : 로그함수로 변환
# step_other(Neighborhood, threshold = 0.01) : 값이 적은 항목을 기타로 변환
# step_upsample(diagnosis) # 데이터 균형화
# step_pca: 주성분분석

lr_recipe <- train_data %>%
  recipe(가격 ~ .) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_pca(all_numeric_predictors(), num_comp = tune())       # pca tuning

# 확인
lr_recipe %>%
  tidy()

# 변수 역활 확인
lr_recipe %>%
  summary()

# recipe 변환 확인
lr_recipe %>%
  prep() %>%
  juice() %>%
  str()






# 7.Model 만들기(모델 설정)

# 모델 인자(argument) 확인
# install.packages("glmnet")                 # glmnet 설치
args(linear_reg) 

# 모델 세팅
lr_model <- linear_reg(penalty = tune(),  
                       mixture = 1) %>% 
  set_engine("glmnet") %>%                 # lm -> glmnet 수정
  set_mode("regression")

# 하이퍼파라미터 그리드 만들기
lr_grid <-expand.grid(penalty = 10^seq(-4, -1, length.out = 5),    # pca tuning
                      num_comp = 2:5)
lr_grid






# 8.workflow 만들기
lr_workflow <- workflow() %>% 
  add_model(lr_model) %>%
  add_recipe(lr_recipe)   

lr_workflow






# 9.Model 훈련(모델 학습)

# 교차검증(k-fold) 데이터 만들기
set.seed(123)
housing_folds <- vfold_cv(train_data, v = 10)

housing_folds

# 기존모델
#lr_fit <- lr_workflow %>%    # workflow수정
#  fit(data = train_data)

# 모델 훈련
lr_results <- lr_workflow %>% 
  tune_grid(resamples = housing_folds,
            grid = lr_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(rmse,mae, rsq))

lr_results

# 튜닝 결과 확인
lr_results %>% 
  collect_metrics()

# 튜닝 결과 그래프 그리기
autoplot(lr_results)                # 튜닝결과 그래프 보기 추가

# best 튜닝 확인
lr_results %>%
  show_best("rsq", n=10) %>%
  arrange(desc(mean))

lr_results %>%
  show_best("rmse", n=10) %>%
  arrange(desc(mean))

# best model 선택
lr_best <- lr_results %>%
  select_best("rsq")

lr_best

# workflow에 best model 파라미터 세팅
final_workflow <- lr_workflow %>%    # 기본 workflow에 세팅
  finalize_workflow(lr_best)         # 기본 workflow에 세팅

final_workflow






# 10.모델 test 및 평가
# 기존모델
# lr_fit %>%
#   predict(test_data)

# 모델 테스트
final_fit <- final_workflow %>%  # final_workflow 수정
  last_fit(housing_split)      

final_fit

# 데이터 예측
final_pred <- final_fit %>%
  collect_predictions()       # collect_predictions() 사용

final_pred

# 모델 평가
# rmse, mae, rsq(r2)

metrics <- metric_set(rmse, mae, rsq)

final_pred %>%
  metrics(truth = 가격, 
          estimate = .pred)

# 데이터 그래프

final_pred %>%                    # final_pred 수정
  ggplot(aes(x = 가격, 
             y = .pred)) + 
  geom_abline(color = "gray50", 
              lty = 2) + 
  geom_point(alpha = 0.5) + 
  coord_obs_pred() + 
  labs(x = "observed", 
       y = "predicted")

# 중요변수 확인
final_fit %>%
  extract_fit_engine() %>%
  vip(num_features = 10)












