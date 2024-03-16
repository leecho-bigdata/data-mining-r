# Ch1101.연관분석






# 1.package 설치
# install.packages("tidyverse") 
# install.packages("arules")
# install.packages("arulesViz")
library(tidyverse)
library(arules)
library(arulesViz)






# 2.데이터 불러오기
book_tb <- read_csv('CharlesBookClub.csv',
                    col_names = TRUE,
                    locale=locale('ko', encoding='euc-kr'),
                    na=".") %>% # csv 데이터 읽어오기
  mutate_if(is.character, as.factor)

str(book_tb)
book_tb






# 3.전처리작업: dataframe을 matrix 형식으로 변환

# 필요없는 변수 제거
book_count <- book_tb %>%
  select (c(8:18))

book_count

# 1이상인 데이터를 1로 변환
# matrix형태로 자동 변환

book_count <- ifelse(book_count > 0, 1, 0)

book_count






# 4.바이너리 코드를 transactions으로 변환
book_trans <- book_count %>%
  as("transactions")

inspect(book_trans)






# 5.연관규칙 실행

# 변수 중요도 그래프 확인
itemFrequencyPlot(book_trans, 
                  support = 0.1,
                  topN = 10)

# apriori(data, minimum support, minimum confidence, and target)
book_rules <- apriori(book_trans,
                      parameter = list(supp= 200/4000, 
                                       conf = 0.5,
                                       target = "rules"))

book_rules






# 6.규칙확인: lift가 높은 순서로 sorting

# rules을 테이블 형태로 저장
rules_tbl <- book_rules %>%
  inspect() %>%
  as_tibble(.name_repair = "unique")

rules_tbl

# 특정 조건에 맞는 규칙 찾기
rules_tbl %>%
  arrange(desc(lift)) %>% 
  filter (support >= 0.1, 
          confidence >= 0.5,
          lift >=1)

# 특정 items 찾기
rules_tbl %>%
  arrange(desc(lift)) %>% 
  filter(rhs == "{YouthBks}")






# 7.그래프 그리기
plot(sort(book_rules, 
          by = "support"), 
     method = "grouped")

# {item} → {item}: 지지도(support)
# color: 향상도(Lift)
plot(sort(book_rules, 
          by = "lift")[1:20], 
     method = "graph", 
     engine = "htmlwidget")















