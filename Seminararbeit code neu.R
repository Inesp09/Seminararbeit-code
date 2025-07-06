#-------------------------------------------------------------------------------

rm(list = ls())

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# loading Packages
#-------------------------------------------------------------------------------
library(tidyverse)
library(car)
library(corrplot)
library(factoextra)
library(cluster)
library(caret)
library(randomForest)
library(pROC)
library(pscl)
library(knitr)

# ------------------------------------------------------------------------------
# Loading and Cleaning Data
# ------------------------------------------------------------------------------
# Reading Data
setwd("/Users/inespalzer/Documents/Uni/6. Semester/Seminar/Seminar/Daten")
bmi <- read_tsv("estat_hlth_ehis_bm1e.tsv")       # BMI
physact <- read_tsv("estat_hlth_ehis_pe2e.tsv")   # Physical activity
fruit <- read_tsv("estat_hlth_ehis_fv3e.tsv")     # Consumption of fruits and vegetables
smoke <- read_tsv("estat_hlth_ehis_sk1e.tsv")     # Smoking
alcohol <- read_tsv("estat_hlth_ehis_al1e.tsv")   # Alcohol consumption

# Cleaning EHIS-Data
clean_ehis <- function(df, var_names, year_filter = "2019") {
  first_column <- names(df)[1]
  year_columns <- setdiff(names(df), first_column)
  df_long <- pivot_longer(df, cols = all_of(year_columns), names_to = "year", values_to = "value")
  df_separated <- separate(df_long, col = first_column, into = var_names, sep = ",", fill = "right")
  df_separated$geo <- str_remove(df_separated$geo, "\\\\TIME_PERIOD")
  df_filtered <- df_separated[df_separated$year == year_filter, ]
  df_filtered$value <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", df_filtered$value)))
  df_final <- df_filtered[!is.na(df_filtered$value), ]
  return(df_final)
}

# Clean Data
bmi_clean <- clean_ehis(bmi, c("freq", "unit", "bmi", "isced11", "sex", "age", "geo"))
physact_clean <- clean_ehis(physact, c("freq", "unit", "physact", "isced11", "sex", "age", "geo"))
fruit_clean <- clean_ehis(fruit, c("freq", "unit", "fruit", "isced11", "sex", "age", "geo"))
smoke_clean <- clean_ehis(smoke, c("freq", "unit", "smoke", "isced11", "sex", "age", "geo"))
alcohol_clean <- clean_ehis(alcohol, c("freq", "unit", "alcohol", "isced11", "sex", "age", "geo"))

# ------------------------------------------------------------------------------
# Correlation Analysis
# ------------------------------------------------------------------------------
# BMI greater than 25
bmi_data <- bmi_clean %>% filter(age == "Y25-34", bmi == "BMI_GE25") %>%
  dplyr::select(geo, sex, isced11, bmi = value)

# smoking daily
smoke_data <- smoke_clean %>% filter(age == "Y25-34", smoke == "SM_DAY") %>%
  dplyr::select(geo, sex, isced11, smoke = value)

# no pysical activity 
activity_data <- physact_clean %>% filter(age == "Y25-34", physact %in% c("MN0")) %>%
  dplyr::select(geo, sex, isced11, physact = value)

# daily alcohol consumption
alcohol_data <- alcohol_clean %>% filter(age == "Y25-34", alcohol == "DAY") %>%
  dplyr::select(geo, sex, isced11, alcohol = value)

# no portion fruit and vegetables
fruit_data <- fruit_clean %>% filter(age == "Y25-34", fruit == "0") %>%
  dplyr::select(geo, sex, isced11, fruit = value)

# combining data
merged_data <- list(bmi_data, smoke_data, activity_data, alcohol_data, fruit_data) %>%
  reduce(left_join, by = c("geo", "sex", "isced11")) %>%
  drop_na()

# correlation plot
cor_matrix <- merged_data %>% select(bmi, smoke, physact, alcohol, fruit) %>%
  cor(method = "spearman")
corrplot(cor_matrix, method = "color", addCoef.col = "black",
         tl.col = "black", mar = c(0,0,2,0))
title("Health Risk Behavior Correlations", line = 3)

# ------------------------------------------------------------------------------
# K-Means & PCA
# ------------------------------------------------------------------------------
# scaling data
scaled_data <- merged_data %>% select(bmi, smoke, physact, alcohol, fruit) %>%
  scale()

# elbow method 
fviz_nbclust(scaled_data, kmeans, method = "wss") + 
  labs(title = "Elbow Method to Determine Optimal Number of Clusters")

# Sensitivity analysis for k = 2
kmeans_result_k2 <- kmeans(scaled_data, centers = 2, nstart = 25)
pca <- prcomp(scaled_data)

set.seed(123)
pca_df_k2 <- as.data.frame(pca$x[, 1:2]) %>%
  mutate(cluster = factor(kmeans_result_k2$cluster))

ggplot(pca_df_k2, aes(PC1, PC2, color = cluster)) +
  geom_point(size = 2) +
  labs(title = "K-Means Clustering with k = 2")

# PCA for k = 3
set.seed(123)
kmeans_result <- kmeans(scaled_data, centers = 3, nstart = 25)

summary(pca)
round(pca$rotation, 2)

pca_df <- as.data.frame(pca$x[, 1:2]) %>%
  mutate(cluster = factor(kmeans_result$cluster))

ggplot(pca_df, aes(PC1, PC2, color = cluster)) +
  geom_point(size = 2) +
  labs(title = "K-Means Clustering - PCA Projection")

# Calculating the mean of each center
cluster_centers <- aggregate(scaled_data, by = list(cluster = kmeans_result$cluster), FUN = mean)
kable(cluster_centers, caption = "Means of Standardized Variables per K-Means Cluster")

# ------------------------------------------------------------------------------
# Hierachical Clustering
# ------------------------------------------------------------------------------
hc_dist <- dist(scaled_data)

hc_ward <- hclust(hc_dist, method = "ward.D2")

fviz_dend(hc_ward, k = 3, rect = TRUE, main = "Ward.D2 Linkage")

# ------------------------------------------------------------------------------
# Data cleaning for Logististic Regression model and Random Forest model
# ------------------------------------------------------------------------------
logit_data <- merged_data %>%
  mutate(region = case_when(
    geo %in% c("AT", "BE", "CH", "DE", "DK", "FI", "FR", "IE", "IS", "LU", "NL", "NO", "SE", "UK") ~ "NorthWest",
    geo %in% c("BG", "CY", "CZ", "EE", "EL", "ES", "HR", "HU", "IT", "LT", "LV", "MT", "PL", "PT", "RO", "RS", "SI", "SK", "TR") ~ "SouthEast"
  )) %>%
  filter(!is.na(region)) %>%
  mutate(edu_binary = ifelse(isced11 %in% c("ED0-2", "ED3_4"), "low", "high"))

# calculation of 75% quantile and risk threshold of 2 
thresholds <- logit_data %>% summarise(across(bmi:fruit, ~quantile(.x, 0.75)))
logit_data <- logit_data %>%
  mutate(risk = factor(
    if_else(
      rowSums(across(bmi:fruit, ~ .x > thresholds[[cur_column()]])) >= 2,
      "risk", "no_risk"
    ),
    levels = c("no_risk", "risk")
  ))

# ------------------------------------------------------------------------------
# Splitting Data
# ------------------------------------------------------------------------------
set.seed(2025)
train_index <- createDataPartition(logit_data$risk, p = 0.8, list = FALSE)
train_data <- logit_data[train_index, ]
test_data  <- logit_data[-train_index, ]

# Distribution of training data
ggplot(train_data, aes(x = risk, fill = risk)) +
  geom_bar() +
  labs(title = "Class distribution in training data",
       x = "Risk class", y = "Count") +
  scale_fill_manual(values = c("skyblue", "tomato")) +
  theme_minimal()

# ------------------------------------------------------------------------------
# Logististic Regression model
# ------------------------------------------------------------------------------
# Logistic Regression model
logit_model <- glm(risk ~ edu_binary + sex + region, data = train_data, family = binomial)
summary(logit_model)

# Pseudo-R^2
pR2(logit_model)

# VIF
vif(logit_model)

# ROC
pred_probs <- predict(logit_model, newdata = test_data, type = "response")
logit_roc <- roc(test_data$risk, pred_probs)
plot(logit_roc, col = "blue", lwd = 2, legacy.axes = FALSE, 
     main = paste("ROC – Logistic Model\nAUC =", round(auc(logit_roc), 3)))

# best cutoff with Youden's statistc
cutoff <- coords(logit_roc, x = "best", best.method = "youden")[[1]]
pred_class <- factor(ifelse(pred_probs > cutoff, "risk", "no_risk"),
                     levels = c("no_risk", "risk"))

# confusion matrix
conf_mat <- table(Predicted = pred_class, Actual = test_data$risk)
print(conf_mat)

# calculate metrics
TP <- conf_mat["risk", "risk"]
FP <- conf_mat["risk", "no_risk"]

accuracy    <- mean(pred_class == test_data$risk)
sensitivity <- sensitivity(pred_class, test_data$risk, positive = "risk")
specificity <- specificity(pred_class, test_data$risk, negative = "no_risk")
ppv         <- TP / (TP + FP)

cat("Accuracy:", round(accuracy, 3), "\n")
cat("Sensitivity:", round(sensitivity, 3), "\n")
cat("Specificity:", round(specificity, 3), "\n")
cat("PPV:", round(ppv, 3), "\n")

# ------------------------------------------------------------------------------
# Random Forest model - no cross-validation
# ------------------------------------------------------------------------------
# conversion to factors
rf_data <- logit_data %>%
  dplyr::select(risk, edu_binary, sex, region) %>%
  mutate(across(c(edu_binary, sex, region), factor))  

# Random Forest model
set.seed(2025)
rf_model <- randomForest(risk ~ edu_binary + sex + region, data = rf_data, ntree = 500, importance = TRUE)
print(rf_model)

# OOB Error
plot(rf_model, main = "Out-of-Bag Error vs Number of Trees")

# confusion matrix
rf_conf <- rf_model$confusion
rf_TN <- rf_conf["no_risk", "no_risk"]
rf_FP <- rf_conf["no_risk", "risk"]
rf_FN <- rf_conf["risk", "no_risk"]
rf_TP <- rf_conf["risk", "risk"]

# calculate metric
rf_accuracy     <- (rf_TP + rf_TN) / sum(rf_conf)
rf_sensitivity  <- rf_TP / (rf_TP + rf_FN)  
rf_specificity  <- rf_TN / (rf_TN + rf_FP) 
rf_ppv          <- rf_TP / (rf_TP + rf_FP) 

cat("Accuracy:     ", round(rf_accuracy, 3), "\n")
cat("Sensitivity:  ", round(rf_sensitivity, 3), "\n")
cat("Specificity:  ", round(rf_specificity, 3), "\n")
cat("PPV:", round(rf_ppv, 3), "\n")

# ROC Curve
rf_probs <- predict(rf_model, newdata = rf_data, type = "prob")[, "risk"]
roc_rf <- roc(response = rf_data$risk, predictor = rf_probs, levels = c("no_risk", "risk"), direction = "<")
plot(roc_rf, legacy.axes = FALSE, col = "darkgreen", lwd = 2,
     main = paste("ROC – Random Forest\nAUC =", round(auc(roc_rf), 3)))

# plot variable importance
varImpPlot(rf_model, main = "Variable Importance – Random Forest", n.var = 3)

# ------------------------------------------------------------------------------
# Random Forest model - with cross-validation
# ------------------------------------------------------------------------------
# cross-validation
set.seed(2025)
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE,
                     summaryFunction = twoClassSummary, savePredictions = "final")

# Random Forest
rf_cv <- train(risk ~ edu_binary + sex + region, data = train_data,
               method = "rf", trControl = ctrl, metric = "ROC", importance = TRUE)

# predicting data
rf_test_probs <- predict(rf_cv, newdata = test_data, type = "prob")[, "risk"]
rf_test_pred  <- predict(rf_cv, newdata = test_data)

# ROC
rf_cv_roc <- roc(response = test_data$risk, predictor = rf_test_probs, levels = c("no_risk", "risk"))
plot(rf_cv_roc, col = "purple", lwd = 2, main = paste("ROC – Random Forest (CV)\nAUC =", round(auc(rf_cv_roc), 3)))

# confusion matrix
confusionMatrix(rf_test_pred, test_data$risk, positive = "risk")

# Plot variable importance
var_imp <- varImp(rf_cv)
var_imp$importance <- var_imp$importance[complete.cases(var_imp$importance), , drop = FALSE]
plot(var_imp, top = 3, main = "Variable Importance - Random Forest (CV)")

# ------------------------------------------------------------------------------
# XGBoost model - with cross-validation
# ------------------------------------------------------------------------------
# define a grid of hyperparameters
xgb_grid <- expand.grid(nrounds = c(100, 200), max_depth = c(3, 6), eta = c(0.01, 0.05, 0.1), 
                        gamma = 0, colsample_bytree = 0.8, min_child_weight = 1, 
                        subsample = 0.8)

# 5-fold-cross-validation
set.seed(2025)
xgb_model <- train(risk ~ edu_binary + sex + region, data = train_data, 
                   method = "xgbTree", trControl = ctrl,
                   tuneGrid = xgb_grid, metric = "ROC")

# prediction on test data
xgb_results <- xgb_model$results
xgb_test_probs <- predict(xgb_model, newdata = test_data, type = "prob")[, "risk"]

# sensitivity analysis
ggplot(xgb_results, aes(x = factor(max_depth), y = factor(eta), fill = ROC)) +
  geom_tile() +
  geom_text(aes(label = round(ROC, 3)), color = "white", size = 3.5) +
  facet_wrap(~ nrounds) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "AUC sensitivity across max_depth and eta by nrounds",
       x = "max_depth", y = "eta", fill = "AUC")

plot(xgb_model)
print(xgb_model)

# ROC curve
xgb_cv_roc <- roc(response = test_data$risk, predictor = xgb_test_probs, levels = c("no_risk", "risk"))
plot(xgb_cv_roc, col = "blue", lwd = 2, main = paste("ROC – XGBoost (CV)\nAUC =", 
                                                     round(auc(xgb_cv_roc), 3)))
# confusion matrix     
xgb_test_pred  <- predict(xgb_model, newdata = test_data)
confusionMatrix(xgb_test_pred, test_data$risk, positive = "risk")
