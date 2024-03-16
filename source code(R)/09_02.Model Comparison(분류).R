# 09_02.여러 모델 비교(분류)
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
breast_tb <- read_csv('유방암진단.csv', 
                      col_names = TRUE,
                      locale=locale('ko', encoding='euc-kr'), # 한글
                      na=".") %>% # csv 데이터 읽어오기
  mutate_if(is.character, as.factor)

str(breast_tb)
breast_tb






# 3.data 전처리

# 필요없는 변수제거
# recipe에서 제거할 수도 있음
breast_tb <- breast_tb %>%
  select(-c(id))  

str(breast_tb)
breast_tb

# 범주형 변수(factor) 변환
# 결과변수(class)에서 관심있는 변수를 1번으로 세팅
breast_tb <- breast_tb %>%
  mutate(진단 = factor(진단, 
                     levels = c(1, 0),              #관심변수=Yes           
                     labels = c("악성", "양성")))






# 4.데이터 탐색(EDA)

# 데이터 탐색: 범주형, 연속형 구분
# skimr::skim() - package명을 앞에 써서 구분
# 패키지를 여러개 사용할 경우에 이름이 같은 경우도 있어서
# 구분이 필요할 경우에 [패키지명::]을 사용
breast_tb %>%
  skimr::skim() 

breast_tb %>%
  group_by(진단) %>%
  skimr::skim() 

# base accuracy
breast_tb %>% 
  count(진단) %>% 
  mutate(prop = n/sum(n))

# 그래프로 확인
breast_tb %>%
  select(진단, 반지름:프랙탈) %>%
  ggpairs(columns = 2:10, 
          mapping = aes(color = 진단, 
                        alpha = 0.5))






# 5.훈련용, 테스트용 데이터 분할: partition

# 데이터 partition
set.seed(123) # 시드 고정 

# 기본 75%는 훈련용, 25%는 테스트용으로 구분
# 결과변수 비율 반영
breast_split <- breast_tb %>%
  initial_split(strata = 진단) # 결과변수 비율반영

breast_split

# training, test용 분리
train_data <- breast_split %>%
  training()

test_data  <- breast_split %>%
  testing()

train_data %>% 
  count(진단) %>% 
  mutate(prop = n/sum(n))

test_data %>% 
  count(진단) %>% 
  mutate(prop = n/sum(n))






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

# 기본 레시피
basic_recipe <- train_data %>%
  recipe(진단 ~ .) %>%
  step_zv(all_predictors()) %>%
  step_smote(진단, over_ratio = 0.75)  # step_smote는 더미변수로 처리
# step_smote는 범주형 자료를 더미변수로 처리하지 않으면 에러남
# 순서가 중요함: step_dummy -> step_smote

# linear형 모델 레시피
linear_recipe <- basic_recipe %>%
  step_normalize(all_numeric()) %>%
  step_dummy(all_nominal_predictors())

# tree형 모델 레시피
tree_recipe <- basic_recipe %>% 
  step_dummy(all_nominal_predictors())

# 확인
linear_recipe %>%
  tidy()

# 변수 역활 확인
linear_recipe %>%
  summary()

# recipe 변환 확인
linear_recipe %>%
  prep() %>%
  juice()

tree_recipe %>%
  prep() %>%
  juice()

tree_recipe %>%
  prep() %>%
  juice() %>%
  count(진단)%>%          
  mutate(prop = n/sum(n))






# 7.Model 만들기(모델 설정)

# classification
lrc_model <- logistic_reg(penalty = tune(), # 모델의 복잡성(0~1)
                          mixture = 1) %>%  # 0:ridge, 1:lasso
  set_engine("glmnet") %>%
  set_mode("classification")
lrc_model
# neural networks
annc_model <- mlp(hidden_units = tune(),
                  penalty = tune(),
                  epochs = 100) %>%          # default = 100  
  set_engine("nnet") %>%
  set_mode("classification")
annc_model
# support vector machine
svmc_model <- svm_rbf(cost = tune(), 
                      rbf_sigma = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")
svmc_model
# decision tree
dtc_model <- decision_tree(cost_complexity = tune(),
                           tree_depth = tune(),
                           min_n = tune()) %>%  
  set_engine("rpart") %>%  
  set_mode("classification")
dtc_model
# random forest
cores <- parallel::detectCores()

rfc_model <- rand_forest(mtry = tune(),  # 투입되는 설명변수 수
                         min_n = tune(), # 노드에 포함되는 최소 데이터 수
                         trees = 100) %>% # begging 갯수(tree수) 
  set_engine("ranger",
             importance = "impurity",   # importance 출력
             num.threads = cores) %>% 
  set_mode("classification")
rfc_model
# XGBoosting tree
xgbc_model <- boost_tree(tree_depth = tune(), # 최대 트리 깊이
                         min_n = tune(),
                         trees = 100) %>% #boost 갯수(tree수) 
  set_engine("xgboost", 
             num.threads = cores) %>% 
  set_mode("classification")
xgbc_model
# K-nearest neighbors 
knnc_model <- nearest_neighbor(neighbors = tune(),
                               dist_power = tune(), 
                               weight_func = tune()) %>%  
  set_engine("kknn") %>%                 
  set_mode("classification")
knnc_model
# discriminant
library(discrim)

dac_model <- discrim_linear(penalty = tune()) %>% 
  set_engine("mda") %>%                 
  set_mode("classification")
dac_model
# naive Bayes
nbc_model <- naive_Bayes(smoothness = tune(),
                         Laplace = tune()) %>% 
  set_engine("naivebayes") %>%                 # lm -> glmnet 수정
  set_mode("classification")
nbc_model





# 8.workflow 만들기

# 선형 모델
linear_workflow <- workflow_set(
  preproc = list(linear = linear_recipe), 
  models = list(lr  = lrc_model, 
                ann = annc_model,
                svm = svmc_model,
                knn = knnc_model,
                da  = dac_model,
                nb = nbc_model)
)

linear_workflow

linear_workflow %>% 
  extract_workflow(id = "linear_ann")

# ann 파리미터 수정
nnet_param <- annc_model %>% 
  extract_parameter_set_dials() %>% 
  update(hidden_units = hidden_units(c(1, 27))) # 27개까지만 가능

linear_workflow <- linear_workflow %>% 
  option_add(param_info = nnet_param, 
             id = "linear_ann")

# 트리 모델
tree_workflow <- workflow_set(
  preproc = list(tree = tree_recipe), 
  models = list(dt = dtc_model, 
                rf = rfc_model, 
                bt = xgbc_model)
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
breast_folds <- vfold_cv(train_data, v = 10)

breast_folds

# 모델 훈련
final_results <- all_workflows %>%
  workflow_map(seed = 123,
               resamples = breast_folds,
               grid = 10,
               control = control_grid(save_pred = TRUE,
                                      parallel_over = "everything",
                                      save_workflow = TRUE),
               verbose = TRUE)

final_results

# 튜닝 결과 확인
final_results %>% 
  rank_results() %>% 
  filter(.metric == "roc_auc") %>% 
  select(wflow_id, model, .config, roc_auc = mean, rank)

# 튜닝 결과 그래프 그리기
final_results %>%
  autoplot(rank_metric = "roc_auc",
           metric = "roc_auc",
           select_best = TRUE) +
  geom_text(aes(y = mean - 1/70, 
                label = wflow_id), 
            hjust = 1) +
  lims(y = c(0.8, 1)) +
  theme(legend.position = "none")

# 개별 모델 튜닝 결과 그래프 그리기
autoplot(final_results, 
         id = "ann", 
         metric = "roc_auc")

# best model 선택
final_best <- final_results %>% 
  extract_workflow_set_result("ann") %>% 
  select_best(metric = "roc_auc")

final_best

# workflow에 best model 파라미터 세팅
final_workflow <- final_results %>% 
  extract_workflow("ann") %>%           # 최종 workflow 추출
  finalize_workflow(final_best)

final_workflow






# 10.모델 test 및 평가

# 모델 테스트
final_fit <- final_workflow %>%  # final_workflow 수정
  last_fit(breast_split)      

final_fit

# 데이터 예측
final_pred <- final_fit %>%
  collect_predictions()       # collect_predictions() 사용

final_pred

# 정오분류표(confusion matrix) 만들기
final_conf <- final_pred  %>%
  conf_mat(truth = 진단, 
           estimate = .pred_class)

final_conf

# 정오분류표 그래프 그리기
autoplot(final_conf, type = "heatmap") # mosaic
autoplot(final_conf, type = "mosaic")

# 모델 평가표 요약
# f1: 재현율(Recall)(↑)과 정밀도(Precision)(↑)
# 재현율(Recall): 실제 Class 중에 잘 맞춘 것(=TPR=민감도)
# 정밀도(Precision): 예측 Class 중에 잘 맞춘 것
# 정확도 (Accuracy) : 클래스 0과 1 모두를 정확하게 분류

summary(final_conf)

# ACU(area under the curve): ROC 정확도

final_pred %>%
  roc_auc(truth = 진단, 
          .pred_악성)

# ROC 커브
final_pred %>%
  roc_curve(truth = 진단, 
            .pred_악성) %>%
  autoplot()

# lift 커브
final_pred %>%
  lift_curve(truth = 진단, 
             .pred_악성) %>%
  autoplot()

# gain 커브
final_pred %>%
  gain_curve(truth = 진단, 
             .pred_악성) %>%
  autoplot()

# 중요변수 확인
final_fit %>%
  extract_fit_engine() %>%
  vip(num_features = 10)













































