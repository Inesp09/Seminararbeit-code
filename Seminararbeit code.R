#------------------------------------------------------------------------------

rm(list = ls())

#------------------------------------------------------------------------------

# ---------------------------
# Packages
# ---------------------------

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

# ---------------------------
# 1. DATA LOADING AND CLEANING
# ---------------------------
# Reading Data
setwd("C:\\Users\\ipalz\\OneDrive\\Dokumente\\Uni\\statistik\\6. Semester\\Seminar")

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

# CLean Data
bmi_clean <- clean_ehis(bmi, c("freq", "unit", "bmi", "isced11", "sex", "age", "geo"))
physact_clean <- clean_ehis(physact, c("freq", "unit", "physact", "isced11", "sex", "age", "geo"))
fruit_clean <- clean_ehis(fruit, c("freq", "unit", "fruit", "isced11", "sex", "age", "geo"))
smoke_clean <- clean_ehis(smoke, c("freq", "unit", "smoking", "isced11", "sex", "age", "geo"))
alcohol_clean <- clean_ehis(alcohol, c("freq", "unit", "frequenc", "isced11", "sex", "age", "geo"))

# ---------------------------
# CORRELATION ANALYSIS
# ---------------------------
# BMI greater than 25
bmi_data <- bmi_clean %>% filter(age == "Y25-34", bmi == "BMI_GE25") %>%
  dplyr::select(geo, sex, isced11, bmi = value)

# smoking daily
smoke_data <- smoke_clean %>% filter(age == "Y25-34", smoking == "SM_DAY") %>%
  dplyr::select(geo, sex, isced11, smoke = value)

# no pysical activity 
activity_data <- physact_clean %>% filter(age == "Y25-34", physact %in% c("MN0")) %>%
  dplyr::select(geo, sex, isced11, physact = value)

# daily alcohol consumption
alcohol_data <- alcohol_clean %>% filter(age == "Y25-34", frequenc == "DAY") %>%
  dplyr::select(geo, sex, isced11, alcohol = value)

# no portion fruit and vegetables
fruit_data <- fruit_clean %>% filter(age == "Y25-34", fruit == "0") %>%
  dplyr::select(geo, sex, isced11, fruit = value)

# combining data
data <- bmi_data %>%
  left_join(smoke_data,   by = c("geo", "sex", "isced11")) %>%
  left_join(activity_data,by = c("geo", "sex", "isced11")) %>%
  left_join(alcohol_data, by = c("geo", "sex", "isced11")) %>%
  left_join(fruit_data,   by = c("geo", "sex", "isced11"))

# correlation plot 
correlation_matrix_spearman <- data %>% 
  dplyr::select(bmi, smoke, physact, alcohol, fruit) %>%
  cor(method = "spearman", use = "complete.obs")

corrplot(correlation_matrix_spearman, method = "color", addCoef.col = "black",
         tl.col = "black", mar = c(0,0,2,0))
title("Health Risk Behavior Correlations", line = 3)

# ---------------------------
# K-MEANS CLUSTERING
# ---------------------------
# Selecting and cleaning relevant data
health <- c("bmi", "smoke", "physact", "alcohol", "fruit")
cluster_data <- data %>%  drop_na(all_of(health)) %>% dplyr::select(all_of(health))

# Standardize variables
cluster_scaled <- scale(cluster_data)

# Determine optimal number of clusters - Elbow method
fviz_nbclust(cluster_scaled, kmeans, method = "wss") + 
  labs(title = "Elbow Method to Determine Optimal Number of Clusters")

# Applying k-Means
set.seed(123)
kmeans_result <- kmeans(cluster_scaled, centers = 3, nstart = 25)

# PCA
pca_kmeans <- prcomp(cluster_scaled)
pca_df <- as.data.frame(pca_kmeans$x[, 1:2]) %>% mutate(cluster = factor(kmeans_result$cluster))

# Visulazation of PCA
pca_var <- pca_kmeans$sdev^2
pca_var_perc <- round(pca_var / sum(pca_var) * 100, 1)
pca_var_perc[1:2]  # zeigt % der Varianz für PC1 und PC2

ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "K-Means Clustering - PCA",
       x = paste0("PC1 (", pca_var_perc[1], "%)"),
       y = paste0("PC2 (", pca_var_perc[2], "%)")) +
  theme_minimal()

# Calculating the mean of each center
cluster_centers <- aggregate(cluster_scaled, by = list(cluster = kmeans_result$cluster), FUN = mean)
kable(cluster_centers, caption = "Means of Standardized Variables per K-Means Cluster")

# ---------------------------
# HIERACHICAL CLUSTERING
# ---------------------------
hc_dist <- dist(cluster_scaled)

hc_single <- hclust(hc_dist, method = "single")
hc_complete <- hclust(hc_dist, method = "complete")
hc_average  <- hclust(hc_dist, method = "average")
hc_ward     <- hclust(hc_dist, method = "ward.D2")

fviz_dend(hc_single, k = 3, rect = TRUE, main = "Single Linkage")
fviz_dend(hc_complete, k = 3, rect = TRUE, main = "Complete Linkage")
fviz_dend(hc_average,  k = 3, rect = TRUE, main = "Average Linkage")
fviz_dend(hc_ward,     k = 3, rect = TRUE, main = "Ward.D2 Linkage")

# ---------------------------
# DATA CLEANING FOR LOGISTIC REGRESSION
# ---------------------------
# Merge all health behavior indicators
data_list <- list(bmi_data, smoke_data, activity_data, alcohol_data, fruit_data)

logit_data <- reduce(data_list, left_join, by = c("geo", "sex", "isced11")) %>%
  drop_na(bmi, smoke, alcohol, physact, fruit, isced11)

# Define regions
logit_data <- logit_data %>%
  mutate(
    region = case_when(
      geo %in% c("AT", "BE", "CH", "DE", "DK", "FI", "FR", "IE", "IS", "LU", "NL", "NO", "SE", "UK") ~ "NorthWest",
      geo %in% c("BG", "CY", "CZ", "EE", "EL", "ES", "HR", "HU", "IT", "LT", "LV", "MT", "PL", "PT", "RO", "RS", "SI", "SK", "TR") ~ "SouthEast"
    )
  ) %>%
  filter(!is.na(region)) 

# calculate upper quantiles
bmi_q3     <- quantile(logit_data$bmi,     0.75, na.rm = TRUE)
smoke_q3   <- quantile(logit_data$smoke,   0.75, na.rm = TRUE)
alcohol_q3 <- quantile(logit_data$alcohol, 0.75, na.rm = TRUE)
physact_q3 <- quantile(logit_data$physact, 0.75, na.rm = TRUE)
fruit_q3   <- quantile(logit_data$fruit,   0.75, na.rm = TRUE)


# create risk_strict 
logit_data <- logit_data %>%
  mutate(
    risk = ifelse(
      (bmi > bmi_q3) + (smoke > smoke_q3) + (alcohol > alcohol_q3) +
        (physact > physact_q3) + (fruit > fruit_q3) >= 2,
      1, 0
    )
  )

# Define binary education variable
logit_data <- logit_data %>%
  mutate(edu_binary = ifelse(isced11 %in% c("ED0-2", "ED3_4"), "lower", "high"))

# ---------------------------
# DATA SPLIT
# ---------------------------
# Spliting data in 80% training and 20% testing
set.seed(2025)
train_index <- createDataPartition(logit_data$risk, p = 0.8, list = FALSE)
train_data <- logit_data[train_index, ]
test_data  <- logit_data[-train_index, ]

# ---------------------------
# LOGISTIC REGRESSION
# ---------------------------
# Logistic regression model on training data
logit_model <- glm(risk ~ edu_binary + sex + region, family = binomial, 
                   data = logit_data)
summary(logit_model)

# Pseudo-R^2
pR2(logit_model)

# VIF
vif(logit_model)

# ROC curve
pred_probs <- predict(logit_model, newdata = test_data, type = "response")
logit_roc <- roc(test_data$risk, pred_probs)
plot(logit_roc, col = "blue", lwd = 2, legacy.axes = FALSE, 
     main = paste("ROC – Logistic Model\nAUC =", round(auc(logit_roc), 3)))

# best cutoff with Youden's statistc
cutoff <- coords(logit_roc, x = "best", best.method = "youden")[[1]]
pred_class <- ifelse(pred_probs > cutoff, 1, 0)

# confusion matrix
conf_mat <- table(Predicted = pred_class, Actual = test_data$risk)
print(conf_mat)

# calculate metrics
TP <- conf_mat["1", "1"]
FP <- conf_mat["1", "0"]

accuracy <- mean(pred_class == test_data$risk)
sensitivity <- sensitivity(factor(pred_class), factor(test_data$risk), positive = "1")
specificity <- specificity(factor(pred_class), factor(test_data$risk), negative = "0")
ppv <- TP / (TP + FP)

cat("Accuracy:", round(accuracy, 3), "\n")
cat("Sensitivity:", round(sensitivity, 3), "\n")
cat("Specificity:", round(specificity, 3), "\n")
cat("PPV:", round(ppv, 3), "\n")

# ---------------------------
# RANDOM FOREST & XGBOOST 
# ---------------------------
# filtering soziodemographic predictors
rf_data <- logit_data %>%
  dplyr::select(risk, edu_binary, sex, region)

# Ensure categorical variables are factors
rf_data$edu_binary <- factor(rf_data$edu_binary)
rf_data$sex <- factor(rf_data$sex)
rf_data$region <- factor(rf_data$region)

# Design matrix
X_rf <- model.matrix(~ ., data = rf_data[,-1])[,-1]
y_rf <- as.factor(rf_data$risk)

# ---------------------------
# RANDOM FOREST MODEL - NO CROSSVALIDATION
# ---------------------------
# random Forest model with 500 trees
set.seed(2025)
rf_model <- randomForest(x = X_rf, y = y_rf, ntree = 500, importance = TRUE)
print(rf_model)

# confusion matrix
rf_conf <- rf_model$confusion
rf_TN <- rf_conf["0", "0"]
rf_FP <- rf_conf["0", "1"]
rf_FN <- rf_conf["1", "0"]
rf_TP <- rf_conf["1", "1"]

# calculate metric
rf_accuracy     <- (rf_TP + rf_TN) / sum(rf_conf)
rf_sensitivity  <- rf_TP / (rf_TP + rf_FN)  
rf_specificity  <- rf_TN / (rf_TN + rf_FP) 
rf_ppv          <- rf_TP / (rf_TP + rf_FP) 

cat("Accuracy:     ", round(rf_accuracy, 3), "\n")
cat("Sensitivity:  ", round(rf_sensitivity, 3), "\n")
cat("Specificity:  ", round(rf_specificity, 3), "\n")
cat("PPV:", round(rf_ppv, 3), "\n")

# predicting class probability
rf_probs <- predict(rf_model, type = "prob")[,2]

# ROC curve
roc_rf <- roc(response = rf_data$risk, predictor = rf_probs)
auc_rf <- auc(roc_rf)
plot(roc_rf, legacy.axes = FALSE, col = "darkgreen", lwd = 2,
     main = paste("ROC – Random Forest\nAUC =", round(auc_rf, 3)))


# plot variable importance
varImpPlot(rf_model, main = "Variable Importance – Random Forest")

# ---------------------------
# RANDOM FOREST WITH CROSS-VALIDATION
# ---------------------------
# recoding outcome variables
set.seed(2025)
train_data$risk <- factor(train_data$risk, labels = c("no_risk", "risk"))
test_data$risk  <- factor(test_data$risk,  labels = c("no_risk", "risk"))

# cross-validation
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE,
                     summaryFunction = twoClassSummary, savePredictions = "final")

# Train cross-validated
rf_cv <- train(risk ~ edu_binary + sex + region, data = train_data, method = "rf", 
               trControl = ctrl, metric = "ROC",
               importance = TRUE)
print(rf_cv)

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

# ---------------------------
# XGBoost MODEL - CROSSVALIDATION
# ---------------------------
# define a grid of hyperparameters
xgb_grid <- expand.grid(nrounds = c(100, 200), max_depth = c(3, 6), eta = c(0.01, 0.05, 0.1), 
                        gamma = 0, colsample_bytree = 0.8, min_child_weight = 1, 
                        subsample = 0.8)

# 5-fold-cross-validation
set.seed(2025)
xgb_model <- train(risk ~ edu_binary + sex + region, data = train_data, 
                   method = "xgbTree", trControl = ctrl,
                   tuneGrid = xgb_grid, metric = "ROC")
plot(xgb_model)
print(xgb_model)

# prediction on test data
xgb_test_probs <- predict(xgb_model, newdata = test_data, type = "prob")[, "risk"]
xgb_test_pred  <- predict(xgb_model, newdata = test_data)


# ROC curve
xgb_cv_roc <- roc(response = test_data$risk, predictor = xgb_test_probs, levels = c("no_risk", "risk"))
plot(xgb_cv_roc, col = "blue", lwd = 2, main = paste("ROC – XGBoost (CV)\nAUC =", 
                                                     round(auc(xgb_cv_roc), 3)))

# confusion matrix                
confusionMatrix(xgb_test_pred, test_data$risk, positive = "risk")


