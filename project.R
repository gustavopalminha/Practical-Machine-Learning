########################################################
#init...
setwd("D:\\Formação\\Coursera\\Practical Machine Learning\\Project-Writeup")
set.seed(123)

########################################################
#submission function to write answers as txt files for each test case.
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

########################################################
#libraries to be loaded...
library("caret")
library("randomForest")

#######################################################
#reading data (train and test)...
#training
training_original <- read.csv(file="pml-training.csv", head=TRUE, sep=",", na.strings = c("NA", "","#DIV/0!"))
#summary(training_original)

#testing...
testing_original <- read.csv(file="pml-testing.csv", head=TRUE, sep=",", na.strings = c("NA", "","#DIV/0!"))
#summary(testing_original)

########################################################
#columns to be dropped after summary of the data which contains NAs and dummy text like DIV/0...
divCols <- c("kurtosis_roll_belt","kurtosis_picth_belt","kurtosis_yaw_belt","skewness_roll_belt","skewness_yaw_belt","amplitude_yaw_belt","kurtosis_yaw_arm","skewness_roll_arm","skewness_pitch_arm","skewness_yaw_arm","kurtosis_roll_dumbbell","kurtosis_yaw_dumbbell","skewness_roll_dumbbell","skewness_yaw_dumbbell","amplitude_yaw_dumbbell","kurtosis_roll_forearm","kurtosis_picth_forearm","kurtosis_yaw_forearm","skewness_roll_forearm","skewness_pitch_forearm","skewness_yaw_forearm","max_yaw_forearm","min_yaw_forearm","amplitude_yaw_forearm")
nasCols <- c("kurtosis_roll_arm", "kurtosis_picth_arm", "var_yaw_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm", "stddev_yaw_arm", "avg_roll_arm", "stddev_roll_arm", "var_roll_arm", "avg_pitch_arm", "var_accel_arm", "stddev_yaw_belt", "var_yaw_belt", "avg_pitch_belt", "stddev_pitch_belt", "var_pitch_belt", "avg_yaw_belt", "var_total_accel_belt", "avg_roll_belt", "stddev_roll_belt", "var_roll_belt", "min_pitch_belt", "min_yaw_belt", "amplitude_roll_belt", "amplitude_pitch_belt", "max_roll_belt", "max_picth_belt", "max_yaw_belt", "min_roll_belt", "skewness_roll_belt.1", "kurtosis_picth_dumbbell", "skewness_pitch_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell", "avg_pitch_dumbbell", "amplitude_pitch_forearm", "max_roll_arm","max_picth_arm","max_yaw_arm","min_roll_arm","min_pitch_arm","min_yaw_arm","amplitude_roll_arm","amplitude_pitch_arm","amplitude_yaw_arm","roll_dumbbell","pitch_dumbbell","yaw_dumbbell","max_roll_dumbbell","max_picth_dumbbell","min_roll_dumbbell","min_pitch_dumbbell","amplitude_roll_dumbbell","amplitude_pitch_dumbbell","var_accel_dumbbell","avg_roll_dumbbell","stddev_roll_dumbbell","var_roll_dumbbell","stddev_pitch_dumbbell","var_pitch_dumbbell","avg_yaw_dumbbell","stddev_yaw_dumbbell","var_yaw_dumbbell","max_roll_forearm","max_picth_forearm","min_roll_forearm","min_pitch_forearm","amplitude_roll_forearm","var_accel_forearm","avg_roll_forearm","stddev_roll_forearm","var_roll_forearm","avg_pitch_forearm","stddev_pitch_forearm","var_pitch_forearm","avg_yaw_forearm","stddev_yaw_forearm","var_yaw_forearm")
otherCols <- c("raw_timestamp_part_1", "raw_timestamp_part_2","cvtd_timestamp", "new_window", "num_window", "user_name", "problem_id", "X")
colsSkip <- c(divCols , nasCols, otherCols)

########################################################
#filter training and testing with columns to be dropped
#1st... training
training <- training_original[, !(names(training_original) %in% colsSkip)]
training$classe <- as.factor(training$classe)
#summary(training)

#...then testing...
trainingCols <- names(training)
testing <- testing_original[,names(testing_original) %in% trainingCols]
#summary(testing)

########################################################
#cross validation using known data (training)... 
#split training into 60 vs 40% for training >> training and training >> testing
trainingSubSet <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
trainingTrainSubSet <- training[trainingSubSet, ]
trainingTestSubSet <- training[-trainingSubSet, ]

#compute train using random forests.. and predict
trainingSubSetRF <- randomForest(classe~., data= trainingTrainSubSet)
trainingTestSubSetPR <- predict(trainingSubSetRF, trainingTestSubSet)

#confusion table results...
trainingConfTable <- confusionMatrix(trainingTestSubSetPR, trainingTestSubSet$classe)
trainingConfTableAll <- confusionMatrix(trainingTestSubSetPR, trainingTestSubSet$classe)$overall


########################################################
#build models using predictor random forests...
modelFitrForest <- randomForest(classe ~ ., data = training, importance = T)
#get the features variance of importance... 
modelFitrForestVarImpObj <- varImp(modelFitrForest)
#and plot it...
varImpPlot(modelFitrForest, sort=TRUE)

########################################################
#compute predictions with random forests...
resultrForest <- predict(modelFitrForest, testing)
#pml_write_files(resultrForest)


########################################################
# APAGAR CODIGO EM BAIXO
########################################################

?varImp

modelFitrForest$results[2,2] * 100

varImpObj <- varImp(modelFitrForest)
varImpPlot(modelFitrForest, sort=TRUE)

plot(varImpObj, main = "Variable Importance of Top 52", top =5)

missClass = function(values, prediction) {
    sum(prediction != values)/length(values)
}
errRate <- missClass(trainingTrainSubSet$classe, trainingSubSetRF)


########################################################
#confusionMatrix(training_prediction,training$classe)
#testing_prediction <- predict(modelFitrForest, newdata=testing)
#confusionMatrix(testing_prediction,testing$classe)


trainingSubSetRF <- randomForest(classe~., data= trainingTrainSubSet)
trainingTestSubSetPR <- predict(trainingSubSetRF, trainingTestSubSet)
confusionMatrix(trainingTestSubSetPR, trainingTestSubSet$classe)
confusionMatrix(trainingTestSubSetPR, trainingTestSubSet$classe)$overall

training_prediction <- predict(trainingSubSetRF, newdata=trainingTrainSubSet)
confusionMatrix(training_prediction,trainingTrainSubSet$classe)

confusionMatrix(table(trainingTestSubSetPR, trainingTestSubSet$classe))
confusionMatrix(table(trainingTestSubSetPR, trainingTestSubSet$classe))$overall

?confusionMatrix

summary(modelFitrPart)
modelFitrForest$confusion
#modelFitrForest <- train(classe ~ ., data = training, method="rf", importance = T) #randomForest(classe ~ ., data = training, importance = T)