# 10_2.군집분석
# https://www.tidymodels.org/learn/statistics/k-means/






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

utilities_tb <- read_csv('Utilities.csv',
                         col_names = TRUE,
                         locale=locale('ko', encoding='euc-kr'),
                         na=".") %>% # csv 데이터 읽어오기
  mutate_if(is.character, as.factor)

str(utilities_tb)
utilities_tb






# 3.데이터 탐색(EDA)

# 데이터 분포 확인
utilities_tb %>%
  ggplot(mapping = aes(x = Sales, 
                       y = Fuel_Cost)) +
  geom_point()

# 그래프로 확인
utilities_tb %>%
  select(Fixed_charge:Fuel_Cost) %>%
  ggpairs()






# 4.데이터 정규화
# 데이터 정규화: mutate_if, 수치형 변수만 정규화
# 회사이름을 row 이름으로 변경

utilities_tb <- utilities_tb %>%
  mutate_if(is.numeric, funs(scale(.))) %>%
  column_to_rownames(var = "Company")

utilities_tb






# 5.최적 군집수 찾기: 엘보우(Elbow) 챠트

# 군집 9개
kclusts <- tibble(k = 2:9) %>%
  mutate(kclust = map(k, ~kmeans(utilities_tb, .x)),
         tidied = map(kclust, tidy),
         glance = map(kclust, glance),
         augmented = map(kclust, augment, utilities_tb))

kclusts

# 개별 행에 대한 값
assignments <- kclusts %>% 
  unnest(cols = c(augmented))

assignments

# 개별 클러스트에 대한 값
clusters <- kclusts %>%
  unnest(cols = c(tidied))

clusters

# 전체 클러스트에 대한 값
clusterings <- kclusts %>%
  unnest(cols = c(glance))

clusterings

# 엘보우(Elbow) 챠트
clusterings %>%
  ggplot(mapping = aes(x = k, 
                       y = tot.withinss)) +
  geom_line() +
  geom_point()

# 군집별 그래프

assignments %>%
  ggplot(mapping = aes(x = Sales,
                       y = Fuel_Cost)) +
  geom_point(mapping = aes(color = .cluster),
             alpha = 0.5) +
  facet_wrap( ~ k) +
  geom_point( data = clusters,
              size = 5,
              shape = "x")






# 6.best K-mean clustering 결과

# best model 구축

set.seed(123)

kclust_best <- kmeans(utilities_tb, 
         centers = 3)

# 군집분석 결과 확인

tidy(kclust_best)






# 7.군집별 특성 파악

# 군집분석 결과 확인
kclust_cl <- tidy(kclust_best) %>%
  select(-c(size, withinss)) %>%
  pivot_longer(c("Fixed_charge", #c("1999, 2000")에러남
                 "Cost",
                 "RoR",
                 "Load_factor",
                 "Demand_growth",
                 "Sales",
                 "Nuclear",
                 "Fuel_Cost"),  
               names_to = "type",
               values_to = "mean")

kclust_cl

# 군집별 특성 그래프
kclust_cl %>%
  ggplot(mapping = aes(x = type,
                       y = mean,
                       group = cluster,
                       color = cluster)) +
  geom_line()

















