# load data
train <-  read.csv("/Users/ningliu/Desktop/STATS-503-Project-main/train.csv")
test <-  read.csv("/Users/ningliu/Desktop/STATS-503-Project-main/test.csv")

# delete 1st and 2nd col
train <- train[,3:dim(train)[2]]
test <- test[,3:dim(test)[2]]

# convert categorical variables into numeric variables
train$Location = as.factor(train$Location)
train$WindGustDir = as.factor(train$WindGustDir)
train$WindDir9am = as.factor(train$WindDir9am)
train$WindDir3pm = as.factor(train$WindDir3pm)
train$RainToday = as.factor(train$RainToday)
train$RainTomorrow <- ifelse(train$RainTomorrow=="Yes",1,0)

test$Location = as.factor(test$Location)
test$WindGustDir = as.factor(test$WindGustDir)
test$WindDir9am = as.factor(test$WindDir9am)
test$WindDir3pm = as.factor(test$WindDir3pm)
test$RainToday = as.factor(test$RainToday)
test$RainTomorrow <- ifelse(test$RainTomorrow=="Yes",1,0)


set.seed(503)
library(mvtnorm)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(randomForest)
library(gbm)


Kfold_boosting <- function(K, tree, depth, shrink, train){
  fold_size = floor(nrow(train)/K)
  cv_error = rep(0,K)
  for(i in 1:K){
    if(i!=K){
      CV_test_id = ((i-1)*fold_size+1):(i*fold_size)
    }else{
      CV_test_id = ((i-1)*fold_size+1):nrow(train)
    }
    CV_train = train[-CV_test_id,]
    CV_test = train[CV_test_id,]
    
    # fit boosting
    model_boost = gbm(RainTomorrow ~., data = CV_train, distribution = "adaboost", n.trees = tree,
                       interaction.depth = depth, shrinkage = shrink)
    pred_cv_test = predict(model_boost, CV_test, n.tree = tree, type="response")
    pred = ifelse(pred_cv_test > 0.5, 1, 0)
    cv_error[i] = mean(pred != CV_test$RainTomorrow)
  }
  return(mean(cv_error))
}


# best depth
K_fold = 5
depth = c(5,6,7,8,9,10)
cv_error_depth= rep(0,length(depth))
for (i in 1:length(depth)){
  cv_error_depth[i] = Kfold_boosting(K = K_fold, tree = 100, depth = depth[i], shrink = 0.1, train)
}
min(cv_error_depth)
best_depth = which(cv_error_depth == min(cv_error_depth))
best_depth
# the 3th in best_depth, which is best_depth=7


# best shrinkage
shrink = seq(0.01,0.19,0.02)
cv_error_shrink = rep(0,length(shrink))
for (i in 1:length(shrink)){
  cv_error_shrink[i] = Kfold_boosting(K = K_fold, tree = 100, depth = 3, shrink = shrink[i], train)
}
min(cv_error_shrink)
best_shrink= which(cv_error_shrink == min(cv_error_shrink))
best_shrink
# the 7th in best_shrink, which is best_depth=0.13


# best tree
tree = seq(1000,2500,500)
cv_error_tree = rep(0,length(tree))
for (i in 1:length(tree)){
  cv_error_tree[i] = Kfold_boosting(K = K_fold, tree = tree[i], depth = 3, shrink = 0.1, train)
}
min(cv_error_tree)
best_tree = which(cv_error_tree == min(cv_error_tree))
best_tree
# the 3rd in best_tree, which is best_tree=2000


# boosting with best parameters
ada_best = gbm(RainTomorrow~., data = train, distribution = "adaboost", n.trees = 2000, interaction.depth = 7, shrinkage = 0.13)
summary(ada_best)


# AUC & BER & testing error
library(ROCR)
library(Metrics)
library(measures)
pred_prob_bst = predict(ada_best, test, type = "response")
pred_bst = ifelse(pred_prob_bst>0.5,1,0)
pr_bst <- prediction(pred_prob_bst, test$RainTomorrow)
AUC_bst = performance(pr_bst, measure= 'auc')
BER_bst = BER(test$RainTomorrow, pred_bst)
AUC_bst@y.values
BER_bst
# AUC_best = 0.9300877
# BER_best = 0.2093104
testing_error = mean(pred_bst != test$RainTomorrow)
testing_error
# testing_error = 0.1187819