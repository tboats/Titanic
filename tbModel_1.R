## load packages
library(ggplot2)
library(randomForest)
library(caret)
library(wesanderson)
library(viridis)
library(plotly)
library(dplyr)
library(plotrix)

## define functions of interest
acc <- function(pred, actual){
    acc <- sum(pred==actual)/length(pred)
    return(acc)
}

## set the working directory
if (Sys.info()['sysname'] == 'Darwin'){
    setwd("~/Dropbox/Kaggle/Titanic")
} else{
    setwd("E:/Dropbox/Kaggle/Titanic")
}
# setwd("~/Dropbox/Kaggle/Titanic")

## load in the data
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)
genderClassModel <- read.csv("genderclassmodel.csv", stringsAsFactors = FALSE)
genderModel <- read.csv("gendermodel.csv")

#################################
## kaggle starting point
set.seed(1)
extractFeatures <- function(data) {
    features <- c("Pclass",
                  "Age",
                  "Sex",
                  "Parch",
                  "SibSp",
                  "Fare",
                  "Embarked")
    fea <- data[,features]
    fea$Age[is.na(fea$Age)] <- -1
    fea$Fare[is.na(fea$Fare)] <- median(fea$Fare, na.rm=TRUE)
    fea$Embarked[fea$Embarked==""] = "S"
    fea$Sex      <- as.factor(fea$Sex)
    fea$Embarked <- as.factor(fea$Embarked)
    return(fea)
}

rf <- randomForest(extractFeatures(train), as.factor(train$Survived), ntree=100, importance=TRUE)

submission <- data.frame(PassengerId = test$PassengerId)
submission$Survived <- predict(rf, extractFeatures(test))
write.csv(submission, file = "1_random_forest_r_submission.csv", row.names=FALSE)

imp <- importance(rf, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
    geom_bar(stat="identity", fill="#53cfff") +
    coord_flip() + 
    theme_light(base_size=20) +
    xlab("") +
    ylab("Importance") + 
    ggtitle("Random Forest Feature Importance\n") +
    theme(plot.title=element_text(size=18))

ggsave("2_feature_importance.png", p)
# end of kaggle starting point
#########################################

## search for best parameters of random forest

# split into training and validation
trainIndex <- createDataPartition(train$Survived, p = 0.8, list=FALSE, times=1 )
tr <- (train[trainIndex,])
val <- (train[-trainIndex,])

# iterate through parameters
treeSizes <- c(100,400)
mns <- c(3,5,10,20,40,100,200,400,401) # max nodes
ms <- c(1,2)
accs <- matrix(0, length(mns), length(treeSizes))

for(tridx in 1:length(treeSizes)){
    for(msidx in 1:length(mns)){
        rf <- randomForest(extractFeatures(tr), as.factor(tr$Survived), ntree=treeSizes[tridx], mtry=2, importance=TRUE, maxnodes = mns[msidx], nodeSize=100*max(mns))
        accs[msidx, tridx] <- acc(predict(rf, extractFeatures(val)),val$Survived)
    }
}


# iterate through parameters after splitting the data set into parts
# iterate through parameters
treeSizes <- c(10,20,30,35,40,42,45,48,50,55,60)
mns <- c(2,4,6,8,10,15,20,25,30) # max nodes
mtrys <- 1
accs <- matrix(0, length(mns), length(treeSizes))

for(tridx in 1:length(treeSizes)){
    for(msidx in 1:length(mns)){
        #trdf <- extractFeatures(tr)
        
        # split into 2 parts based on Sex
        trm <- filter(tr, Sex=="male")
        trf <- filter(tr, Sex=="female")
        valm <- filter(val, Sex=="male")
        valf <- filter(val, Sex=="female")
        
        # train 2 different random forest models
        rfm <- randomForest(extractFeatures(trm), as.factor(trm$Survived), ntree=treeSizes[tridx], mtry=mtrys, importance=TRUE, maxnodes = mns[msidx], nodeSize=100*max(mns))
        rff <- randomForest(extractFeatures(trf), as.factor(trf$Survived), ntree=treeSizes[tridx], mtry=mtrys, importance=TRUE, maxnodes = mns[msidx], nodeSize=100*max(mns))
        
        # recombine results
        finalPrediction <- c(as.numeric(as.character(predict(rfm, extractFeatures(valm)))), as.numeric(as.character(predict(rff, extractFeatures(valf)))))
        actualResult <- c(valm$Survived, valf$Survived)
        accs[msidx, tridx] <- acc(finalPrediction,actualResult)
    }
}
accs
color2D.matplot(accs)
maxLoc <- which(accs == max(accs), arr.ind = TRUE)
treeSizes[maxLoc[,2]]
mns[maxLoc[,1]]
max(accs)
# BEST PARAMETERS:
#   maxnodes = 40
#   ntree = 400
#   mtry = 1
#   accuracy = 86.0%

#   maxnodes = 8
#   ntree = 50
#   mtry = 3
#   accuracy = 87.6%


## train on the full training set

# optimal parameters from validation set tests
ntreeOpt <- 50
mtryOpt <- 3
maxnodesOpt <- 8
# split into 2 parts based on Sex
trainm <- filter(train, Sex=="male")
trainf <- filter(train, Sex=="female")
# train 2 different random forest models
rfm <- randomForest(extractFeatures(trainm), as.factor(trainm$Survived), ntree=ntreeOpt, mtry=mtryOpt, importance=TRUE, maxnodes = maxnodesOpt)
rff <- randomForest(extractFeatures(trainf), as.factor(trainf$Survived), ntree=ntreeOpt, mtry=mtryOpt, importance=TRUE, maxnodes = maxnodesOpt)
# recombine results
finalPrediction <- c(as.numeric(as.character(predict(rfm, extractFeatures(trainm)))), as.numeric(as.character(predict(rff, extractFeatures(trainf)))))
actualResult <- c(trainm$Survived, trainf$Survived)
accTrain <- acc(finalPrediction,actualResult)
accTrain

#######################################
## exploratory plots

# Age, Fare, Relationship
train2 <- train
train2$ageRange <- "young"
ageThresh <- 30
ageOldLoc <- train2$Age > ageThresh
ageOldLoc[is.na(ageOldLoc)] <- FALSE
train2[ageOldLoc, "ageRange"] <- "old"

q <- ggplot(data=train2, aes(x=(jitter(Pclass)), y=Fare, col=as.factor(Survived)))
q <- q + geom_point(size=4, alpha=0.2) + facet_wrap( ageRange ~ Sex) +
    scale_color_manual(values=wes_palette(n=3, name="Darjeeling"), guide = guide_legend(title = "Survived?")) +
    xlab("Passenger Class") +
    scale_x_continuous(breaks=c(1,2,3)) +
    scale_y_log10()
    #scale_x_discrete(breaks=c(1,2,3), labels=c(1,2,3))
print(q)
(qq <- ggplotly(q))
# Fare distribution
train$Survived <- as.factor(train$Survived)
q <- ggplot(data=train, aes(x=Fare, fill=Survived))
q + geom_histogram()

# Guidance:
# try training random forests after splitting up the data set into 2 parts: men and women