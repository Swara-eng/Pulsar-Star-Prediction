################################################################################
#               ___  ________ _____   _________  _____                         #
#               |  \/  |_   _/  ___| |___  /   ||  _  |                        #
#               | .  . | | | \ `--.     / / /| || |_| |                        #
#               | |\/| | | |  `--. \   / / /_| |\____ |                        #
#               | |  | |_| |_/\__/ / ./ /\___  |.___/ /                        #                                        
#               \_|  |_/\___/\____/  \_/     |_/\____/                         #
#                                                                              #
#                                                                              #                                    
#                                                                              #
#      _____            _               _____  _____  _____  _____             #     
#     /  ___|          (_)             / __  \|  _  |/ __  \/ __  \            #           
#     \ `--. _ __  _ __ _ _ __   __ _  `' / /'| |/' |`' / /'`' / /'            #
#      `--. \ '_ \| '__| | '_ \ / _` |   / /  |  /| |  / /    / /              #
#     /\__/ / |_) | |  | | | | | (_| | ./ /___\ |_/ /./ /___./ /___            #
#     \____/| .__/|_|  |_|_| |_|\__, | \_____/ \___/ \_____/\_____/            #
#           | |                  __/ |                                         #
#           |_|                 |___/                                          #
#         _____                     ______          _           _              #
#        |_   _|                    | ___ \        (_)         | |             #
#          | | ___  __ _ _ __ ___   | |_/ / __ ___  _  ___  ___| |_            #
#          | |/ _ \/ _` | '_ ` _ \  |  __/ '__/ _ \| |/ _ \/ __| __|           #
#          | |  __/ (_| | | | | | | | |  | | | (_) | |  __/ (__| |_            #
#          \_/\___|\__,_|_| |_| |_| \_|  |_|  \___/| |\___|\___|\__|           #
#                                                 _/ |                         # 
#                                                |__/                          #    
#               ___  ____ ____ ___  _ ____ ___ _ _  _ ____                     #
#               |__] |__/ |___ |  \ | |     |  | |\ | | __                     #
#               |    |  \ |___ |__/ | |___  |  | | \| |__]                     #
#       ___  _  _ _    ____ ____ ____    ____ ___ ____ ____ ____               #
#       |__] |  | |    [__  |__| |__/    [__   |  |__| |__/ [__                #
#       |    |__| |___ ___] |  | |  \    ___]  |  |  | |  \ ___]               #
################################################################################
############################# Project Team #####################################

#                             Bailey Brown                                     #
#                           Stefanie Dvizac                                   #
#                             Swara Gupta                                      #
#                           Alexander Nestler                                  #

################################################################################

#un-comment below to install corrplot if you don't already have it
##install.packages("corrplot")
###Libraries
library(caret)
library(tree)
library(leaps)
library(e1071)
library(pROC)
library(class)
library(randomForest)
library(MASS)
library(corrplot)

#Clear the environment
rm(list=ls())
#create the testing and training datasets
## Data from: https://raw.githubusercontent.com/alexandrehsd/Predicting-Pulsar-Stars/master/pulsar_stars.csv
df <- read.csv('pulsar_stars.csv')
#set seed!
set.seed(2022)

#Preliminary Analysis
summary(df)
corrplot(cor(df))
pairs(df)

## To create the test and train data sets from data:
##    --First split data into two set:
          #One set with all the observations that ARE a pulsar (target_class = 1)
pulsar.set <- df[df$target_class==1,]
          #One set with all the observations that ARE NOT a pulsar (target_class = 0)
nan.set <- df[df$target_class==0,]
##    --Second split each of these new sets 70/30
          #70/30 split on the set with all pulsar observations
pulsar.n <- nrow(pulsar.set)
pulsar.i <- sample(pulsar.n,0.7*pulsar.n)
pulsar.train <- pulsar.set[pulsar.i,]
pulsar.test <- pulsar.set[-pulsar.i,]
          #70/30 split on the set with observations that are not pulsars
nan.n <- nrow(nan.set)
nan.i <- sample(nan.n,0.7*nan.n)
nan.train <- nan.set[nan.i,]
nan.test <- nan.set[-nan.i,]
##    --Third combine the split data into the final 'test' and 'train' data sets
          #Combine the 70% split data sets from the observations that are pulsars
            # AND that are not pulsars into one data set named 'train'
train <- rbind(pulsar.train,nan.train)
          #Combine the 30% split data sets from the observations that are pulsars
            # AND that are not pulsars into one data set named 'test'
test <- rbind(pulsar.test,nan.test)
################################################################################
##################  MODEL CREATION AND ANALYSIS ################################

##BEGIN BAILEY SECTION (Naive Bayes/KNN)##
##Naive Bayes
#Create Model
nb.fit = naiveBayes(target_class~.,data =df,train)
#Generate Classification with the model on the test data
nb.class = predict(nb.fit,test)
#Summarize results
nb.sum= confusionMatrix(data= as.factor(nb.class), reference=as.factor(test$target_class), positive="1")
#create a raw prediction with the model and the test set for ROC analysis
nb.pred=predict(nb.fit,test, type="raw")
#create ROC and analyze
nb.roc = roc(response=test$target_class,predictor=nb.pred[,1], plot=TRUE, print.auc=TRUE)
##K-nearest neighbors
#iterate through k=1 to k=10, generate knn predictions for each k value model,
# generate confusion matrix for each model created, and store the accuracy for
# each model in a vector for analysis
acc.v <- numeric(10)
for (i in 1:10)
{
  pred <- knn(train, test,train$target_class, k=i)
  cm <- confusionMatrix(data=as.factor(pred), reference=as.factor(test$target_class), positive="1")
  acc.v[i] <- cm$overall[1]
}
#create knn model with the highest accuracy
# NOTE: which.max returns the INDEX LOCATION in the vector that contained the 
#     the highest accuracy, which in this case matches the desired value of k
knn.pred = knn(train,test,train$target_class, k=which.max(acc.v))
knn.sum <- confusionMatrix(data=as.factor(knn.pred), reference=as.factor(test$target_class), positive='1')
########### END BAILEY SECTION ###########

##BEGIN SWARA SECTION (INSERT MODEL TYPES HERE)##
##LDA
#create models, review, plot
lda.fit = lda(target_class~., data = train)
#plot(lda.fit) #warning: possible margins issue!
#generate predictions
lda.pred = predict(lda.fit, test, type="response")
lda.class = lda.pred$class  #prediction. 
#create confusion matrix for analysis
lda.sum= confusionMatrix(data= as.factor(lda.class), reference=as.factor(test$target_class), positive="1")
#sum(lda.pred$posterior[, 1] >= .5)  #use 0.5 as the cut-off value
#sum(lda.pred$posterior[, 1] < .5)
#lda.pred$posterior[1:20, 1]
#lda.class[1:20]
#sum(lda.pred$posterior[, 1] > .9) #use 0.9 as the cut-off value None posterior probability is greater than 0.9!
lda.roc = roc(response= as.factor(test$target_class), predictor=lda.pred$posterior[,1], plot = TRUE, print.auc = TRUE)  #ROC curve
##QDA
###create qda models
qda.fit = qda(target_class~., data = train)
#generate predictions and classification variables
qda.pred = predict(qda.fit, test, type="response")
qda.class = qda.pred$class
#create confusion matrix and ROC for analysis
qda.sum= confusionMatrix(data= as.factor(qda.class), reference=as.factor(test$target_class), positive="1")
qda.roc = roc(response= as.factor(test$target_class), predictor=qda.pred$posterior[,1], plot=TRUE, print.auc=TRUE)
########### END SWARA SECTION ###########

##BEGIN ALEX SECTION (LOGISTIC/CLASS TREE)##
## Logistic Regression ##
glm.model <- glm(target_class~., family='binomial', train)
#get the summary
summary(glm.model)
#two non-significant predictors identified; create new model with those predictors removed
glm2 <- glm(target_class~.-Excess.kurtosis.of.the.DM.SNR.curve -Skewness.of.the.DM.SNR.curve, family='binomial', train)
summary(glm2)
#create probabilities with the created model
glm.probs <- predict(glm2, test, type = "response")
#generate matrix same size as the test set with all '0' values
glm.class <- rep(0,nrow(test))
#create classification set; replace index value in the classification set to '1'
# if the same index value in the probability set is greater than 50%
glm.class[glm.probs > 0.5] <- 1
#generate confusion matrix for analysis
glm.sum <- confusionMatrix(as.factor(glm.class), reference = as.factor(test$target_class), positive = '1')
#generate ROC graph and analyze AUC
glm.roc <- roc(test$target_class~ glm.probs, plot = TRUE, print.auc=TRUE)
## Classification Tree ##
###Change binary values in data set to factors with 1 representing 'Yes' and '0' representing 'No'
# in the training and testing sets
pulsar <- factor(ifelse(train$target_class==1,'Yes','No'))
y.test <- factor(ifelse(test$target_class==1,'Yes','No'))
#create classification tree
tree <- tree(pulsar~.-target_class,train)
#plot the tree
plot(tree)
text(tree,pretty=0)
#summarize results
summary(tree)
#generate classified predictions using the model
tree.pred <- predict(tree, test, type='class')
#obtain prediction probabilities for ROC analysis
tree.pred.prob <- predict(tree,test)
tree.roc = roc(response= y.test, predictor=tree.pred.prob[,2], plot=TRUE, print.auc=TRUE)
#generate confusion matrix for analysis
tree.sum <- confusionMatrix(tree.pred,y.test, positive='Yes')
########### END ALEX SECTION ###########

##BEGIN STEFANIE SECTION (Bagging/Random Forest)##
##Bagging
#to make a classification, not regression version, 
# convert target_class column to factor in the train and test sets
train$target_class = as.factor(train$target_class)
test$target_class = as.factor(test$target_class)
#fill the bag! ALL the way up (all variables used)
bag = randomForest(target_class~., data = train, mtry = 8, importance = TRUE)
#roc for bagging
bag.pred <- predict(bag, test)
bag.roc = roc(response= test$target_class, predictor=as.numeric(bag.pred), plot=TRUE, print.auc=TRUE)
##random Forest
#generate models for cross-validation
rf.1= randomForest(target_class~., data = train, mtry = 1, importance = TRUE)
rf.2= randomForest(target_class~., data = train, mtry = 2, importance = TRUE)
rf.3= randomForest(target_class~., data = train, mtry = 3, importance = TRUE)
rf.4= randomForest(target_class~., data = train, mtry = 4, importance = TRUE)
rf.5= randomForest(target_class~., data = train, mtry = 5, importance = TRUE)
rf.6= randomForest(target_class~., data = train, mtry = 6, importance = TRUE)
rf.7= randomForest(target_class~., data = train, mtry = 7, importance = TRUE)
#calculate OOB error estimate manually for each model
rf1.oobe <- sum(rf.1$err.rate[,1])/nrow(rf.1$err.rate)
rf2.oobe <- sum(rf.2$err.rate[,1])/nrow(rf.2$err.rate)
rf3.oobe <- sum(rf.3$err.rate[,1])/nrow(rf.3$err.rate)
rf4.oobe <- sum(rf.4$err.rate[,1])/nrow(rf.4$err.rate)
rf5.oobe <- sum(rf.5$err.rate[,1])/nrow(rf.5$err.rate)
rf6.oobe <- sum(rf.6$err.rate[,1])/nrow(rf.6$err.rate)
rf7.oobe <- sum(rf.7$err.rate[,1])/nrow(rf.7$err.rate)
#place the values in a vector for analysis
oobe.v <- c(rf1.oobe, rf2.oobe, rf3.oobe, rf4.oobe, rf5.oobe, rf6.oobe, rf7.oobe)
#retrieve the false positive rate/ Type I error for each model
rf1.t1 <- rf.1$confusion[1,3]
rf2.t1 <- rf.2$confusion[1,3]
rf3.t1 <- rf.3$confusion[1,3]
rf4.t1 <- rf.4$confusion[1,3]
rf5.t1 <- rf.5$confusion[1,3]
rf6.t1 <- rf.6$confusion[1,3]
rf7.t1 <- rf.7$confusion[1,3]
#place the values in a vector for analysis
t1.v <- c(rf1.t1, rf2.t1, rf3.t1, rf4.t1, rf5.t1, rf6.t1, rf7.t1)
#retrieve the false negative rate/ Type II error for each model
rf1.t2 <- rf.1$confusion[2,3]
rf2.t2 <- rf.2$confusion[2,3]
rf3.t2 <- rf.3$confusion[2,3]
rf4.t2 <- rf.4$confusion[2,3]
rf5.t2 <- rf.5$confusion[2,3]
rf6.t2 <- rf.6$confusion[2,3]
rf7.t2 <- rf.7$confusion[2,3]
#place the values in a vector for analysis
t2.v <- c(rf1.t2, rf2.t2, rf3.t2, rf4.t2, rf5.t2, rf6.t2, rf7.t2)
#generate roc curves for each model
rf1.pred <- predict(rf.1, test)
rf1.roc = roc(response= test$target_class, predictor=as.numeric(rf1.pred), plot=TRUE, print.auc=TRUE)
rf2.pred <- predict(rf.2, test)
rf2.roc = roc(response= test$target_class, predictor=as.numeric(rf2.pred), plot=TRUE, print.auc=TRUE)
rf3.pred <- predict(rf.3, test)
rf3.roc = roc(response= test$target_class, predictor=as.numeric(rf3.pred), plot=TRUE, print.auc=TRUE)
rf4.pred <- predict(rf.4, test)
rf4.roc = roc(response= test$target_class, predictor=as.numeric(rf4.pred), plot=TRUE, print.auc=TRUE)
rf5.pred <- predict(rf.5, test)
rf5.roc = roc(response= test$target_class, predictor=as.numeric(rf5.pred), plot=TRUE, print.auc=TRUE)
rf6.pred <- predict(rf.6, test)
rf6.roc = roc(response= test$target_class, predictor=as.numeric(rf6.pred), plot=TRUE, print.auc=TRUE)
rf7.pred <- predict(rf.7, test)
rf7.roc = roc(response= test$target_class, predictor=as.numeric(rf7.pred), plot=TRUE, print.auc=TRUE)
########### END STEFANIE SECTION ###########
## Model Summaries ##
print("Logistic Regression Summary:")
glm.sum
print(paste("The Type I error of the logistic regression model:", round(glm.sum$table[1,2]/(glm.sum$table[1,1]+glm.sum$table[1,2]),4)))
print(paste("The Type II error of the logistic regression model:", round(glm.sum$table[2,1]/(glm.sum$table[2,1]+glm.sum$table[2,2]),4)))
print(paste("The misclassification rate of the logistic regression model:",round((glm.sum$table[1,2]+glm.sum$table[2,1])/sum(glm.sum$table),4)))

print("Linear Discriminent Analysis Summary:")
lda.sum
print(paste("The Type I error of the linear discriminant analysis model:", round(lda.sum$table[1,2]/(lda.sum$table[1,1]+lda.sum$table[1,2]),4)))
print(paste("The Type II error of the linear discriminant analysis model:", round(lda.sum$table[2,1]/(lda.sum$table[2,1]+lda.sum$table[2,2]),4)))
print(paste("The misclassification rate of the linear discriminant analysis model:", round((lda.sum$table[1,2]+lda.sum$table[2,1])/sum(lda.sum$table),4)))

print("Quadratic Discriminent Analysis Summary:")
qda.sum
print(paste("The Type I error of the quadratic discriminant analysis model:", round(qda.sum$table[1,2]/(qda.sum$table[1,1]+qda.sum$table[1,2]),4)))
print(paste("The Type II error of the quadratic discriminant analysis model:", round(qda.sum$table[2,1]/(qda.sum$table[2,1]+qda.sum$table[2,2]),4)))
print(paste("The misclassification rate of the quadratic discriminant analysis model:", round((qda.sum$table[1,2]+qda.sum$table[2,1])/sum(qda.sum$table),4)))

print("Naive Bayes Summary:")
nb.sum
print(paste("The Type I error of the naive bayes model:", round(nb.sum$table[1,2]/(nb.sum$table[1,1]+nb.sum$table[1,2]),4)))
print(paste("The Type II error of the naive bayes model:", round(nb.sum$table[2,1]/(nb.sum$table[2,1]+nb.sum$table[2,2]),4)))
print(paste("The misclassification rate of the naive bayes model:",round((nb.sum$table[1,2]+nb.sum$table[2,1])/sum(nb.sum$table),4)))

print("K_Nearest Neighbor Summary:")
knn.sum
print(paste("The Type I error of the KNN model:", round(knn.sum$table[1,2]/(knn.sum$table[1,1]+knn.sum$table[1,2]),4)))
print(paste("The Type II error of the KNN model:", round(knn.sum$table[2,1]/(knn.sum$table[2,1]+knn.sum$table[2,2]),4)))
print(paste("The misclassification rate of the KNN model:",round((knn.sum$table[1,2]+knn.sum$table[2,1])/sum(knn.sum$table),4)))

print("Decision-Tree Summary:")
tree.sum
print(paste("The Type I error of the decision-tree model:", round(tree.sum$table[1,2]/(tree.sum$table[1,1]+tree.sum$table[1,2]),4)))
print(paste("The Type II error of the decision-tree model:", round(tree.sum$table[2,1]/(tree.sum$table[2,1]+tree.sum$table[2,2]),4)))
print(paste("The misclassification rate of the decision-tree model:", round((tree.sum$table[1,2]+tree.sum$table[2,1])/sum(tree.sum$table),4)))

print("Bagging Summary:")
varImpPlot(bag)
print(paste("The Type I error for the bag model is:", round(bag$confusion[1,3],4)))
print(paste("The Type II error for the bag model is:", round(bag$confusion[2,3],4)))
print(paste("The Out of Bag error for the bag model is:", round(sum(bag$err.rate[,1])/nrow(bag$err.rate),4)))

print("Random Forest Summary:")
varImpPlot(rf.1)
varImpPlot(rf.2)
varImpPlot(rf.3)
varImpPlot(rf.4)
varImpPlot(rf.5)
varImpPlot(rf.6)
varImpPlot(rf.7)
print("     Out of Bag Error:")
print(paste("       The average OOB error for all models:", round(mean(oobe.v),4)))
print(paste("       The lowest OOB error among the models:", round(min(oobe.v),4), "with", which.min(oobe.v), "variables."))
print(paste("       The highest OOB error among the models:", round(max(oobe.v),4), "with", which.max(oobe.v), "variables."))
print(paste("       The average Type I error for all models:", round(mean(t1.v),4)))
print(paste("       The lowest Type I error among the models:", round(min(t1.v),4), "with", which.min(t1.v), "variables."))
print(paste("       The highest Type I error among the models:", round(max(t1.v),4), "with", which.max(t1.v), "variables."))
print(paste("       The average Type II error for all models:", round(mean(t2.v),4)))
print(paste("       The lowest Type II error among the models:", round(min(t2.v),4), "with", which.min(t2.v), "variables."))
print(paste("       The highest Type II error among the models:", round(max(t2.v),4), "with", which.max(t2.v), "variables."))
print(paste("       The average AUC for all models: ~92%"))

## ROC/AUC Analysis Results ##
ggroc(list(glm=glm.roc, lda=lda.roc, qda=qda.roc, nb=nb.roc, tree=tree.roc))
print(paste("The AUC for the Logistic Regression model is:", round(auc(glm.roc),4)))
print(paste("The AUC for the Linear Discriminent Analysis model is:", round(auc(lda.roc),4)))
print(paste("The AUC for the Quadratic Discriminent Analysis model is:", round(auc(qda.roc),4)))
print(paste("The AUC for the Naive Bayes model is:", round(auc(nb.roc),4)))
print(paste("The AUC for the Decision-Tree model is:", round(auc(tree.roc),4)))

## Since our data contains more negative examples than positive examples,
##  and since it would probably be more desirable for radio astronomers to
##  reduce the false negative rate (pulsar signal that is labeled as not a pulsar)
##  our focus should be to reduce the false negative rate (which will be the highest
##  between our Type I and Type II errors given the nature of the data as well)
t2.analysis <- c(round(glm.sum$table[2,1]/(glm.sum$table[2,1]+glm.sum$table[2,2]),4), round(lda.sum$table[2,1]/(lda.sum$table[2,1]+lda.sum$table[2,2]),4), round(qda.sum$table[2,1]/(qda.sum$table[2,1]+qda.sum$table[2,2]),4), round(nb.sum$table[2,1]/(nb.sum$table[2,1]+nb.sum$table[2,2]),4), round(knn.sum$table[2,1]/(knn.sum$table[2,1]+knn.sum$table[2,2]),4), round(tree.sum$table[2,1]/(tree.sum$table[2,1]+tree.sum$table[2,2]),4), round(bag$confusion[2,3],4), round(min(t2.v),4))
t2.analysis.models <- c("Logistic Regression", "Linear Discriminent Analysis", "Quadratic Discriminent Analysis", "Naive Bayes", "K-Nearest Neighbor", "Decision-Tree", "Bagging", "Random Forest")
print(paste("The lowest Type II error among all the models created is", t2.analysis[which.min(t2.analysis)], "from the", t2.analysis.models[which.min(t2.analysis)], "model."))
print(paste("The misclassification rate of the linear discriminant analysis model:", round((lda.sum$table[1,2]+lda.sum$table[2,1])/sum(lda.sum$table),4)))
print(paste("The AUC for the Linear Discriminent Analysis model is:", round(auc(lda.roc),4)))
