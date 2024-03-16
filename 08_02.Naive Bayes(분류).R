# 08_02.Naive Bayes(분류)






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
# step_normalize(all_numeric()) : 데이터 정규화
# 특히, penalty 사용하는 모델에서 중요(logistic, SVM 등)
# step_zv(all_predictors()) : 단일 고유 값 (예 : 모두 0) 변수 제거. 
# step_dummy(all_nominal(), -all_outcomes()) : one-hot-ecoding
# step_log(Gr_Liv_Area, base = 10) : 로그함수로 변환
# step_other(Neighborhood, threshold = 0.01) : 값이 적은 항목을 기타로 변환
# step_upsample(diagnosis) # 데이터 균형화
# step_pca: 주성분분석

# 불균형자료 처리
# https://themis.tidymodels.org/reference/step_smote.html

nbc_recipe <- train_data %>%
  recipe(진단 ~ .) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(진단, over_ratio = 0.75)  # step_smote는 더미변수로 처리


# 확인
nbc_recipe %>%
  tidy()

# 변수 역활 확인
nbc_recipe %>%
  summary()

# recipe 변환 확인
nbc_recipe %>%
  prep() %>%
  juice()

nbc_recipe %>%
  prep() %>%
  juice() %>%
  count(진단)%>%          
  mutate(prop = n/sum(n))






# 7.Model 만들기(모델 설정)

# 모델 인자(argument) 확인
# https://parsnip.tidymodels.org/reference/naive_Bayes.html
# install.packages("discrim")
# install.packages("naivebayes")
library("discrim")    # library를 로드해야 함
args(naive_Bayes) 

# 모델 세팅
nbc_model <- naive_Bayes(smoothness = tune(),
                         Laplace = tune()) %>% 
  set_engine("naivebayes") %>%                 # lm -> glmnet 수정
  set_mode("classification")

# 하이퍼파라미터 그리드 만들기
nbc_grid <- grid_regular(smoothness(),   # 최소값과 최대값 설정
                         Laplace(),
                         levels = 5)

nbc_grid






# 8.workflow 만들기
nbc_workflow <- workflow() %>% 
  add_model(nbc_model) %>%
  add_recipe(nbc_recipe)   

nbc_workflow






# 9.Model 훈련(모델 학습)

# 교차검증(k-fold) 데이터 만들기
set.seed(123)
breast_folds <- vfold_cv(train_data, v = 10)

breast_folds

# 모델 훈련
nbc_results <- nbc_workflow %>% 
  tune_grid(resamples = breast_folds,
            grid = nbc_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc, accuracy))   # roc_auc, accuracy

nbc_results

# 튜닝 결과 확인
nbc_results %>% 
  collect_metrics()

# 튜닝 결과 그래프 그리기
autoplot(nbc_results)                # 튜닝결과 그래프 보기 추가

# best 튜닝 확인
nbc_results %>%
  show_best("roc_auc", n=10) %>%
  arrange(desc(mean))

nbc_results %>%
  show_best("accuracy", n=10) %>%
  arrange(desc(mean))

# best model 선택
nbc_best <- nbc_results %>%
  select_best("roc_auc")

nbc_best

# workflow에 best model 파라미터 세팅
final_workflow <- nbc_workflow %>%    # 기본 workflow에 세팅
  finalize_workflow(nbc_best)         # 기본 workflow에 세팅

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
  extract_fit_engine()


































