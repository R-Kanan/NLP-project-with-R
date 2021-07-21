library(text2vec)
library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)
#getting the data read by 'fread' function
df <- fread("C:/Users/ASUS/Downloads/Bootcamp 7/R Week 13/Day 2/emails.csv")
df
#Splitting data
set.seed(123)
split <- df$spam %>% sample.split(SplitRatio = 0.8)
train <- df %>% subset(split == T)
test <- df %>% subset(split == F)
#Tokenizing (separating word by word)
train_tokens <- #train$text %>% tolower() %>% word_tokenizer()
  gsub('[[:punct:]0-9]', ' ',train$text %>% tolower()) %>% word_tokenizer()
#[[:punct:]0-9]
it_train <- train_tokens %>% 
  itoken(progressbar = F)
#Words that do not make a sense to exclude
stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are",'subject','a','s','re',
                'hi')
#Creating vocabulary with 2 and 3 phrases, 
#not just sole words and without stopwords
vocab <- it_train %>% create_vocabulary(ngram = c(2L,3L),stopwords = stop_words)

#Also, keep the input words which matches with the given parameters
pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 1, 
                   doc_proportion_max = 0.8,
                   doc_proportion_min = 0.001)

pruned_vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10) 

#vectorizer and dtm 
vectorizer <- pruned_vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()

#Model building with GLM NET on train data
glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 4,
            thresh = 0.001,
            maxit = 1000)
#AUC score of train data
glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

#Same operations on Test data
it_test <-  
  gsub('[[:punct:]0-9]', ' ',test$text %>% tolower()) %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(
    progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)
#Prediction on Test data
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(2)
