# 09_01.여러 모델 비교(예측)
# https://www.tidymodels.org/start/models/
# https://www.tmwr.org/






# 1.package 설치
# install.packages("tidyverse") 
# install.packages("tidymodels")
# install.packages("skimr")
# install.packages("vip")
# install.packages("GGally")
# install.packages("themis")
library(tidyverse)
library(tidymodels)
library(skimr)           # 데이터 요약(EDA)
library(vip)             # 중요한 변수 찾기
library(GGally)          # 여러 변수 통합 그래프
library(themis)          # 불균형 자료 처리






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
# step_zv(all_predictors()) : 단일 고유 값 (예 : 모두 0) 변수 제거
# step_normalize(all_numeric()) : 데이터 정규화
# 특히, linear형 모델에서 중요(regression,logistic, SVM 등)
# step_dummy(all_nominal(), -all_outcomes()) : one-hot-ecoding
# step_smote: 불균형자료 처리(분류모델에서 사용)
# https://themis.tidymodels.org/reference/step_smote.html
# step_log(Gr_Liv_Area, base = 10) : 로그함수로 변환
# step_other(Neighborhood, threshold = 0.01) : 값이 적은 항목을 기타로 변환
# step_pca: 주성분분석

# tree형 모델 레시피
tree_recipe <- train_data %>%
  recipe(가격 ~ .) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors())

# linear형 모델 레시피
linear_recipe <- tree_recipe %>%
  step_normalize(all_numeric())

# 확인
tree_recipe %>%
  tidy()

# 변수 역활 확인
tree_recipe %>%
  summary()

# recipe 변환 확인
tree_recipe %>%
  prep() %>%
  juice()

linear_recipe %>%
  prep() %>%
  juice()






# 7.Model 만들기(모델 설정)

# regression
lrr_model <- linear_reg(penalty = tune(),  # 모델의 복잡성(0~1)
                        mixture = 1) %>%   # 0:ridge, 1:lasso
  set_engine("glmnet") %>%                 # lm -> glmnet 수정
  set_mode("regression")

# neural networks
annr_model <- mlp(hidden_units = tune(),
                  penalty = tune(),
                  epochs = 100) %>%          # default = 100  
  set_engine("nnet") %>%
  set_mode("regression")
annr_model
# support vector machine
svmr_model <- svm_rbf(cost = tune(), 
                      rbf_sigma = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")
svmr_model
# decision tree
dtr_model <- decision_tree(cost_complexity = tune(),
                           tree_depth = 4,
                           min_n = tune()) %>%  
  set_engine("rpart") %>%  
  set_mode("regression")

# random forest
cores <- parallel::detectCores()

rfr_model <- rand_forest(mtry = tune(),  # 투입되는 설명변수 수
                         min_n = tune(), # 노드에 포함되는 최소 데이터 수
                         trees = 100) %>% # begging 갯수(tree수) 
  set_engine("ranger",
             importance = "impurity",   # importance 출력
             num.threads = cores) %>% 
  set_mode("regression")

# XGBoosting tree
xgbr_model <- boost_tree(tree_depth = tune(), # 최대 트리 깊이
                         min_n = tune(),
                         trees = 100) %>% #boost 갯수(tree수) 
  set_engine("xgboost", 
             num.threads = cores) %>% 
  set_mode("regression")

# K-nearest neighbors 
knnr_model <- nearest_neighbor(neighbors = tune(),
                               dist_power = tune(), 
                               weight_func = tune()) %>%  
  set_engine("kknn") %>%                 # lm -> glmnet 수정
  set_mode("regression")






# 8.workflow 만들기

# 선형 모델
linear_workflow <- workflow_set(
  preproc = list(linear = linear_recipe), 
  models = list(lr = lrr_model, 
                ann = annr_model,
                svm = svmr_model,
                knn = knnr_model)
)

linear_workflow

linear_workflow %>% 
  extract_workflow(id = "linear_ann")

# ann 파리미터 수정
nnet_param <- annr_model %>% 
  extract_parameter_set_dials() %>% 
  update(hidden_units = hidden_units(c(1, 27))) # 27개까지만 가능

linear_workflow <- linear_workflow %>% 
  option_add(param_info = nnet_param, 
             id = "linear_ann")

# 트리 모델
tree_workflow <- workflow_set(
  preproc = list(tree = tree_recipe), 
  models = list(dt = dtr_model, 
                rf = rfr_model, 
                bt = xgbr_model)
)

tree_workflow

tree_workflow %>% 
  extract_workflow(id = "tree_dt")

# 모든 모델 결합
all_workflows <- bind_rows(linear_workflow, 
                           tree_workflow) %>% 
  mutate(wflow_id = gsub("(linear_)|(tree_)", "", wflow_id))

all_workflows






# 9.Model 훈련(모델 학습)

# 교차검증(k-fold) 데이터 만들기
set.seed(123)
housing_folds <- vfold_cv(train_data, v = 10)

housing_folds

# 모델 훈련
final_results <- all_workflows %>%
  workflow_map(seed = 123,
               resamples = housing_folds,
               grid = 10,
               control = control_grid(save_pred = TRUE,
                                      parallel_over = "everything",
                                      save_workflow = TRUE),
               verbose = TRUE)

final_results

# 튜닝 결과 확인
final_results %>% 
  rank_results() %>% 
  filter(.metric == "rsq") %>% 
  select(wflow_id, model, .config, rsq = mean, rank)

final_results %>% 
  rank_results() %>% 
  filter(.metric == "rmse") %>% 
  select(wflow_id, model, .config, rmse = mean, rank)

# 튜닝 결과 그래프 그리기
final_results %>%
  autoplot(rank_metric = "rsq",
           metric = "rsq",
           select_best = TRUE) +
  geom_text(aes(y = mean - 1/30, 
                label = wflow_id), 
            hjust = 1) +
  theme(legend.position = "none")

# 개별 모델 튜닝 결과 그래프 그리기
autoplot(final_results, 
         id = "ann", 
         metric = "rsq")

# best model 선택
final_best <- final_results %>% 
  extract_workflow_set_result("ann") %>% 
  select_best(metric = "rsq")

final_best

# workflow에 best model 파라미터 세팅
final_workflow <- final_results %>% 
  extract_workflow("ann") %>%           # 최종 workflow 추출
  finalize_workflow(final_best)

final_workflow






# 10.모델 test 및 평가

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

#install.packages("NeuralNetTools") 
library(NeuralNetTools)       

# 중요변수 확인
final_fit %>%
  extract_fit_engine() %>%
  vip(num_features = 10)














































