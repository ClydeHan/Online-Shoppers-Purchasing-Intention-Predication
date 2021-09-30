#Problem: Online Shoppers Purchasing In- tention
#The objective is to develop a predictive model for the purchasing intention of an online store. 
#The dataset is composed of 10,000 instances belonging to two classes: 
#False, if the user did not commit the buy; True, if the user finally bought something. 
#There are 17 attributes for each instance. The class is the last one, "Revenue".

#Tasks
#The student has to analyze the data and build effective predic- tive models. 
#All the techniques studied along the course should be applied, namely:
#1. KNN (mknn)
#2. Linear Discriminant Analysis (mlda) 
#3. Logistic Regression (mlr)
#4. Classification trees (mtree)
#5. Random Forests (mrf)
#6. Artificial Neural Networks (mann)
#7. Support Vector Machines (msvm)




## load the dataset
load("/Users/clyde/Downloads/dataset_project.Rdata")
data = osi.train
nrow(data)
ncol(data)
data$Revenue

## add the factor to class Revenue
data$Revenue = as.factor(data$Revenue) # convert class of Revenue to factor
data$Revenue <- factor(data$Revenue, levels= c("TRUE","FALSE")) # change order of factor, our positive class is "TRUE"

## check the variable information 
str(data)

## check the unbalance and distribution
table(data$Revenue)
prop.table(table(data$Revenue))

## convert variable data type which is not in numeric or integer among categorical data to numeric
data$Month <- as.numeric(data$Month)
data$VisitorType <- as.numeric(data$VisitorType)
data$Weekend <- as.numeric(data$Weekend)




## select dataset for training, the remain for testing, 75% for training and 25% for testing
set.seed(2021)
train <- sample(nrow(data),3*nrow(data)/4,replace = FALSE)

data.train = data[train,]
data.test = data[-train,]

table(data.train$Revenue)
table(data.test$Revenue)



## oversampling the minority class and subsampling the majority class to make training dataset balanced
data.train.balanced <- ovun.sample(Revenue ~ ., data = data.train, method = "both", p = 0.4, N = 6000, seed = 100)$data # sampling
data.train.balanced$Revenue <- factor(data.train.balanced$Revenue, levels= c("TRUE","FALSE")) # change order of factor, our positive class is "TRUE"

table(data.train.balanced$Revenue)
prop.table(table(data.train.balanced$Revenue)) # check the balanced training dataset 




## load necessary package
library(ROSE)
library(ROCR)
library(MASS)
library(ISLR)
library(class)
library(tree)
library(rattle)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
library(nnet)

#1. KNN (mknn)

train.knn <- data.train.balanced
test.knn <- data.test
set.seed(20)
mknn <- knn(train.knn[,-18], test.knn[,-18], train.knn$Revenue, k = 5, prob = TRUE) # exclude parameter indicates the class

## evaluation
table(mknn, test.knn$Revenue)
t = table(mknn, test.knn$Revenue)
(t[1,1]+t[2,2])/sum(t) # calculate the accuracy
roc.curve(test.knn$Revenue, mknn) # compute AUC value and plot ROC curve

# compute Precision, Recall, F-score, and Specificity
Precision = t[1,1]/(t[1,1]+t[2,1])
Precision
Recall = t[1,1]/(t[1,1]+t[1,2])
Recall
F_score = (4*Recall*Precision/(4*Recall+Precision))
F_score
Specificity = t[2,2]/(t[2,2]+t[2,1])
Specificity

#2. Linear Discriminant Analysis (mlda) 
mlda <- lda(Revenue ~., data = data.train.balanced)
mlda.pred <- predict(mlda, newdata = data.test, type = "class")

## evaluation
table(mlda.pred$class, data.test$Revenue)
t = table(mlda.pred$class, data.test$Revenue)
(t[1,1]+t[2,2])/sum(t) # calculate the accuracy
roc.curve(data.test$Revenue, mlda.pred$class) # compute AUC value and plot ROC curve

# compute Precision, Recall, F-score, and Specificity
Precision = t[1,1]/(t[1,1]+t[2,1])
Precision
Recall = t[1,1]/(t[1,1]+t[1,2])
Recall
F_score = (4*Recall*Precision/(4*Recall+Precision))
F_score
Specificity = t[2,2]/(t[2,2]+t[2,1])
Specificity


#3. Logistic Regression (mlr)

mlr <- glm(Revenue ~., data = data.train.balanced, family = binomial)
summary(mlr)
mlr.probs <- predict(mlr, newdata = data.test, type = "response" )


contrasts(data$Revenue)

mlr.pred <-ifelse(mlr.probs > 0.5,"FALSE", "TRUE")
mlr.pred = as.factor(mlr.pred)
mlr.pred <- factor(mlr.pred, levels= c("TRUE","FALSE")) # change order of factor, our positive class is "TRUE"

## evaluation
table(mlr.pred, data.test$Revenue)
t = table(mlr.pred, data.test$Revenue)
(t[1,1]+t[2,2])/sum(t) # calculate the accuracy
roc.curve(data.test$Revenue, mlr.pred) # compute AUC value and plot ROC curve

# compute Precision, Recall, F-score, and Specificity
Precision = t[1,1]/(t[1,1]+t[2,1])
Precision
Recall = t[1,1]/(t[1,1]+t[1,2])
Recall
F_score = (4*Recall*Precision/(4*Recall+Precision))
F_score
Specificity = t[2,2]/(t[2,2]+t[2,1])
Specificity

#4. Classification trees (mtree)

mtree <- tree(Revenue ~., data.train.balanced)
#mtree2 <- prune.tree(mtree, best = 7) # in this case, the best result after prune is same as tree(). so we only keep tree()

mtree.pred <- predict(mtree, data.test, type = "class")

## evaluation
table(mtree.pred, data.test$Revenue)
t = table(mtree.pred, data.test$Revenue)
(t[1,1]+t[2,2])/sum(t) # calculate the accuracy
roc.curve(data.test$Revenue, mtree.pred) # compute AUC value and plot ROC curve

# compute Precision, Recall, F-score, and Specificity
Precision = t[1,1]/(t[1,1]+t[2,1])
Precision
Recall = t[1,1]/(t[1,1]+t[1,2])
Recall
F_score = (4*Recall*Precision/(4*Recall+Precision))
F_score
Specificity = t[2,2]/(t[2,2]+t[2,1])
Specificity

## plotting tree structure
fancyRpartPlot(mtree <- rpart(Revenue ~., data.train.balanced))
plot(mtree)
text(mtree)


#5. Random Forests (mrf)

mrf <- randomForest(Revenue ~ ., data = data.train.balanced, mtry = 8, importance = TRUE, proximity = TRUE)
mrf.pred <- predict(mrf, newdata = data.test)

## evaluation
table(mrf.pred, data.test$Revenue)
t = table(mrf.pred, data.test$Revenue)
(t[1,1]+t[2,2])/sum(t) # calculate the accuracy
importance(mrf)
roc.curve(data.test$Revenue, mrf.pred) # compute AUC value and plot ROC curve

# compute Precision, Recall, F-score, and Specificity
Precision = t[1,1]/(t[1,1]+t[2,1])
Precision
Recall = t[1,1]/(t[1,1]+t[1,2])
Recall
F_score = (4*Recall*Precision/(4*Recall+Precision))
F_score
Specificity = t[2,2]/(t[2,2]+t[2,1])
Specificity

#6. Artificial Neural Networks (mann)

mann <- nnet(Revenue~., data = data.train.balanced, size = 9)
summary(mann)

## training 
mann.pred <- predict(mann, newdata = data.train.balanced, type = "class")

mann.pred <- as.factor(mann.pred)
mann.pred <- factor(mann.pred, levels= c("TRUE","FALSE")) # change order of factor, our positive class is "TRUE"

## evaluation
table(mann.pred, data.train.balanced$Revenue)
t = table(mann.pred, data.train.balanced$Revenue)
(t[1,1]+t[2,2])/sum(t) # calculate the accuracy
roc.curve(data.train.balanced$Revenue, mann.pred) # compute AUC value and plot ROC curve

## testing

mann.pred <- predict(mann, newdata = data.test, type = "class")

mann.pred <- as.factor(mann.pred)
mann.pred <- factor(mann.pred, levels= c("TRUE","FALSE")) # change order of factor, our positive class is "TRUE"


## evaluation
table(mann.pred, data.test$Revenue)
t = table(mann.pred, data.test$Revenue)
(t[1,1]+t[2,2])/sum(t) # calculate the accuracy
roc.curve(data.test$Revenue, mann.pred) # compute AUC value and plot ROC curve

# compute Precision, Recall, F-score, and Specificity
Precision = t[1,1]/(t[1,1]+t[2,1])
Precision
Recall = t[1,1]/(t[1,1]+t[1,2])
Recall
F_score = (4*Recall*Precision/(4*Recall+Precision))
F_score
Specificity = t[2,2]/(t[2,2]+t[2,1])
Specificity




#7. Support Vector Machines (msvm)

msvm <- svm(Revenue ~., data = data.train.balanced, kernel= "linear", cost = 1, scale = TRUE)
msvm.pred <- predict(msvm, newdata = data.test, type = "class")

table(msvm.pred, data.test$Revenue)
t = table(msvm.pred, data.test$Revenue)

(t[1,1]+t[2,2])/sum(t) # calculate the accuracy
roc.curve(data.test$Revenue, msvm.pred) # compute AUC value and plot ROC curve

## use tune() to compute best model 
tune.out <- tune(svm, Revenue~., data = data.train.balanced, kernel = "linear", scale = TRUE, cost=c(0.1, 1, 10, 100, 1000)) # compute best model according to cost
summary(tune.out)

msvm.pred <- predict(tune.out$best.model, newdata=data.test)

## evaluation
table(msvm.pred, data.test$Revenue)
t = table(msvm.pred, data.test$Revenue)
(t[1,1]+t[2,2])/sum(t) # calculate the accuracy
roc.curve(data.test$Revenue, msvm.pred) # compute AUC value and plot ROC curve

# compute Precision, Recall, F-score, and Specificity
Precision = t[1,1]/(t[1,1]+t[2,1])
Precision
Recall = t[1,1]/(t[1,1]+t[1,2])
Recall
F_score = (4*Recall*Precision/(4*Recall+Precision))
F_score
Specificity = t[2,2]/(t[2,2]+t[2,1])
Specificity



## final model of each method

mknn
mlda
mlr
mtree
mrf
mann
msvm









