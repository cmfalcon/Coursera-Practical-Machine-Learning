---
title: "Practical Machine Learning Course Project"
author: "Celeste Falcon"
date: "2020/04/17"
output: html_document
editor_options: 
  chunk_output_type: console
---

# Set up, load packages, and record session info
```{r setup.chunk, echo=F}
knitr::opts_chunk$set(warning = F, cache = T, tidy = T)
options(digits = 5)
```

```{r load.packages, cache = F}
# load packages here
library(tidyverse)
library(knitr)
library(conflicted)
library(caret)

# load global functions here
`%nin%` = Negate(`%in%`)

conflict_prefer("filter", "dplyr")
conflict_prefer("select", "dplyr")
```

```{r session_info}
# load Session Info here
sessionInfo()
```

# Data Input

```{r data_input}
training <- read_csv("/gpfs2/duds/giywg/PracticalMachineLearning/project/pml-training.csv")
testing <- read_csv("/gpfs2/duds/giywg/PracticalMachineLearning/project/pml-testing.csv")
```

# Exploratory analysis
I took several steps to better understand the data set itself. 

## Explore data set
These steps included examining the y-variable (classe) and viewing the column names and first 10 observations for each variable (I have commented this out since it creates a very long print out in the html file). I could see that several columns of the test data set were completely missing data (all NA). These will not be useful as predictors and could lead to overfitting of the training data set, so I removed these columns from both the testing and training data.
```{r explore_data_set}
# Explore y-variable to be predicted
unique(training$classe)
class(training$classe)

## Set classe variable to factor in both training data
training$classe <- as.factor(training$classe)

#Examine data variables
# print(training, width = Inf)
# print(testing, width = Inf)

# remove columns that are all NA in testing data set as these will not be useful as predictors and could lead to overfitting in training data set
testing <- janitor::remove_empty(testing, which = "cols") %>% 
  select(-X1)

cols_in_testing <- colnames(testing)

# remove those same columns from training data set so they are not used to create prediction function
training <- training %>% 
  select_if(names(.) %in% cols_in_testing | names(.) == "classe")

```

## Create subset for internal testing from training data set to be able to estimate out of sample error and accuracy
In order to estimate my out of sample error, I decided to further subset the original training data into two groups that I called "internal training" and "internal testing". This allowed me to test prediction models created in "internal training" on the "internal testing" data set and calculate the accuarcy of the method. 
```{r create_internal_train_test_sets}
in_internal_training <- as.vector(createDataPartition(y = training$classe, p = .85, list = FALSE))

internal_training <- training[in_internal_training,]
internal_testing <- training[-in_internal_training,]

internal_testing2 <- internal_testing[,-59]

```

## Exploratory visualizations
I also used visualizations to explore the data. I started by looking at a feature plot. However, with so many predictor variables in this data set, it was difficult to see whether there are any patterns. I then created violin plots for each variable by the classe variable. These plots allowed me to see the distribution of each variable within each classe for quantitative variables, and for qualitative variables it at least showed me whether each level of the variable existed within each classe. For the sake of brevity for my graders, I have not reproduced the violin plots in this html file, but the code for them can be seen in this code chunk. Finally, I created a table of classe by the user_name variable to see whether these were related (plot not shown here to reduce the number of figures graders must review, but code is shown below).
```{r}
featurePlot(x = internal_training[,-59], y = internal_training$classe)

# for (p in 1:(ncol(internal_training) - 1)) {
#   # p=6
#   boxplot <- ggplot(internal_training, aes_string(x = "classe", y = colnames(internal_training)[p])) +
#     geom_violin(draw_quantiles = c(0.25,0.50,0.75)) +
#     ylab(colnames(internal_training)[p])
# 
#   print(boxplot)
# }

t1 <- as.data.frame(table(internal_training$classe, internal_training$user_name))
colnames(t1)[1] <- "classe"

# ggplot(t1, aes(x = classe, y = Freq)) +
#   geom_bar(stat = "identity") +
#   facet_grid(.~Var2)

```


## Exploratory modeling
With so many variables in the data set, I decided to do some exploratory modeling steps to get to know the data better. I started by fitting a classification tree since that would be somewhat interpretable. Timestamp related variables showed up in this tree but my inuition was that these were perhaps not very explanatory and could cause issues as predictors for the final test set. I decided to create several visuals of these timestamp variables to look for patterns, or the absence thereof. Not seeing a pattern between the timestamp variables and classe, I decided to remove these predictors so that more interpretable variables would be used in the prediction model.  I then fitted a new classification tree model without the timestamp variables. The dendogram plot was more interpretable, but the accuracy ot the method was very low.  So, my next step was to try a random forest model to combine by voting across many tree models. In this model, I used 10-fold cross validation to minimize overfitting.
```{r exploratory_modeling}

# Start by fitting a classification tree for initial model exploration since it is fairly interpretable
mod_rpart <- train(classe ~ ., method = "rpart", data = internal_training)
print(mod_rpart$finalModel)

library(rattle)

fancyRpartPlot(mod_rpart$finalModel)

#### Look at classe by timestamp since the timestamp variable come up in the first exploratory model but are difficult to interpret
ggplot(internal_training, aes(x = raw_timestamp_part_1, y = roll_belt, color = user_name)) +
  geom_point()
ggplot(internal_training, aes(x = raw_timestamp_part_2, y = roll_belt, color = user_name)) +
  geom_point()
# ggplot(internal_training, aes(x = raw_timestamp_part_2, y = roll_belt, color = classe)) +
#   geom_point(alpha = 0.4)
# ggplot(internal_training, aes(x = cvtd_timestamp, y = roll_belt, color = user_name)) +
#   geom_point() +
#   theme(axis.text.x = element_text(angle = 45))

# Seeing that the timestamp variables do not seems to correlate with the classe variable, remove these so that more interpretable variables are used in prediction
predictors_to_exclude <- c("raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp")

internal_training2 <- internal_testing %>% 
  select_if(names(.) %nin% predictors_to_exclude)

# Fit a new classification tree model without the timestamp variables as predictors, predict values in the internal testing set, and check accuracy value and confusion matrix
mod_rpart <- train(classe ~ ., method = "rpart", data = internal_training2)
print(mod_rpart$finalModel)

fancyRpartPlot(mod_rpart$finalModel)

pred_rpart <- predict(mod_rpart, newdata = internal_testing)
confusionMatrix(pred_rpart, internal_testing$classe)

# The classification tree model had rather low accuracy, so try random forest model which will combine by voting across many  tree models, predict values in the internal testing set, and check accuracy value and confusion matrix
#I chose to implement cross validation with 10 folds to mitigate overfitting in the training data thereby setting up for a better result in the testing data.
mod_rf <- train(classe ~ ., method = "rf", data = internal_training, trControl = trainControl(method = "cv", number = 10))
pred_rf <- predict(mod_rf, newdata = internal_testing2)
confusionMatrix(pred_rf, internal_testing$classe)

```

This random forest model predicts very well in the internal testing set, so I will use it to predict the classe variable in the final testing set to answer the quiz. Based on the fact that a) I used cross validation to reduce overfitting and b) I had very high accuracy in my internal testing data set, I expect that I will have very low error in predicting in the testing set for the quiz.

# Final model
```{r}
pred_rf_final <- predict(mod_rf, newdata = testing)
pred_rf_final
```

Turning in my answers for the quiz, I found that I got 100% right, verifying that my machine learning prediction method has performed well.
