library(insuranceData)
library(neuralnet)
library(fastDummies)  # For one-hot encoding
library(caret)        # For train-test split
library(fitdistrplus)
library(ggplot2)
library(tidyverse)
library(gridExtra)
library(GGally)
library(reshape2)
library(glm2)
library(forcats)
library(brglm2)
library(e1071)
library(reticulate)  # If needed for NN
library(keras)



data(dataOhlsson)
data = dataOhlsson

str(data)
summary(data)
head(data)

sapply(data, function(x) sum(is.na(x))) # No missing values

# Summary of variable types and example anomalies
summary(data)

# Data PREP
data <- dataOhlsson
data <- data[data$skadkost > 0 & data$duration > 0, ]
set.seed(123)
train_index <- createDataPartition(data$skadkost, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]
vars <- c("skadkost", "agarald", "kon", "zon", "mcklass", "fordald", "bonuskl")
train_model <- train_data %>% select(all_of(vars))
test_model <- test_data %>% select(all_of(vars))
train_model <- model.matrix(skadkost ~ . -1, train_model) %>% as.data.frame()
train_model$skadkost <- train_data$skadkost
test_model <- model.matrix(skadkost ~ . -1, test_model) %>% as.data.frame()
test_model$skadkost <- test_data$skadkost
train_model$log_duration <- log(train_data$duration)
test_model$log_duration <- log(test_data$duration)
normalise_train <- function(x) (x - min(x)) / (max(x) - min(x))
train_features <- train_model %>% select(-skadkost) %>% mutate_all(normalise_train)
mins <- sapply(train_model %>% select(-skadkost), min)
maxs <- sapply(train_model %>% select(-skadkost), max)
normalise_test <- function(x, varname) (x - mins[varname]) / (maxs[varname] - mins[varname])
test_features <- test_model %>% select(-skadkost) %>% mutate(across(everything(), ~ normalise_test(.x, cur_column())))
x_train <- as.matrix(train_features)
y_train <- train_model$skadkost  # Use raw for most models; log if specified
x_test <- as.matrix(test_features)
y_test <- test_model$skadkost

#################################################################################
# Gamma GLM
# Get the design matrix
X <- model.matrix(
  ~ agarald + fordald + bonuskl + kon + zon + mcklass,
  data = train_data
)

# Fit a simple Gamma GLM with log link for starting values
simple_model <- glm(
  skadkost ~ 1,
  family = Gamma(link = "log"),
  data = train_data,
  offset = log(duration),
  control = glm.control(maxit = 100)
)

# Create starting values (intercept from simple model, zeros for others)
start_vals <- c(coef(simple_model), rep(0, ncol(X) - 1))

# Fit Gamma GLM with log link and offset
glm_gamma <- glm2(
  skadkost ~ agarald + fordald + bonuskl + kon + zon + mcklass,
  family = Gamma(link = "log"),
  offset = log(duration),
  data = train_data,
  start = start_vals,
  control = glm.control(maxit = 1000, epsilon = 1e-8)
)

# Summary of the model
summary(glm_gamma)

# Predict on test set
test_data$predicted <- predict(glm_gamma, newdata = test_data, type = "response")

# Evaluate performance on test set
mse_glm <- mean((test_data$skadkost - test_data$predicted)^2)
rmse_glm <- sqrt(mse_glm)
mae_glm <- mean(abs(test_data$skadkost - test_data$predicted))

#################################################################################
# NN
# Select same variables
vars <- c("skadkost", "agarald", "kon", "zon", "mcklass", "fordald", "bonuskl")

train_model <- train_data %>% select(all_of(vars))
test_model <- test_data %>% select(all_of(vars))

# Create design matrices
train_model <- model.matrix(skadkost ~ . -1, train_model) %>% as.data.frame()
train_model$skadkost <- train_data$skadkost

test_model <- model.matrix(skadkost ~ . -1, test_model) %>% as.data.frame()
test_model$skadkost <- test_data$skadkost

# Add log(duration)
train_model$log_duration <- log(train_data$duration)
test_model$log_duration <- log(test_data$duration)

normalise_train <- function(x) (x - min(x)) / (max(x) - min(x))

# Normalise training data
train_features <- train_model %>% select(-skadkost) %>% mutate_all(normalise_train)

# Capture min/max for each variable
mins <- sapply(train_model %>% select(-skadkost), min)
maxs <- sapply(train_model %>% select(-skadkost), max)

# Apply same scaling to test set
normalise_test <- function(x, varname) (x - mins[varname]) / (maxs[varname] - mins[varname])
test_features <- test_model %>%
  select(-skadkost) %>%
  mutate(across(everything(), ~ normalise_test(.x, cur_column())))

x_train <- as.matrix(train_features)
y_train <- log(train_model$skadkost)
y_train_raw <- train_model$skadkost

x_test <- as.matrix(test_features)
y_test <- log(test_model$skadkost)
y_test_raw <- test_model$skadkost

x_train <- np_array(x_train)
y_train <- np_array(y_train)

x_test <- np_array(x_test)
y_test <- np_array(y_test)

gamma_loss <- function(y_true, y_pred){
  K <- backend()
  y_true <- K$clip(y_true, K$epsilon(), K$cast_to_floatx(1e10))
  y_pred <- K$clip(y_pred, K$epsilon(), K$cast_to_floatx(1e10))
  loss <- 2 * ((y_true - y_pred) / y_pred - K$log(y_true / y_pred))
  K$mean(loss, axis = as.integer(-1))
}

# Define the model with the new architecture
model <- keras_model_sequential()
model$add(layer_dense(units = 256, activation = "relu", input_shape = c(ncol(x_train))))
model$add(layer_dropout(rate = 0.2))
model$add(layer_dense(units = 128, activation = "relu"))
model$add(layer_dense(units = 64, activation = "relu"))
model$add(layer_dense(units = 1, activation = "relu"))


# Compile with RMSprop optimizer and specified learning rate
model$compile(
  loss = gamma_loss,
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  metrics = list("mean_squared_error", "mean_absolute_error")
)

# Train the model using batch_size = 128
history <- model$fit(
  x = x_train,
  y = y_train,
  epochs = as.integer(100),
  batch_size = as.integer(128),
  validation_split = as.integer(0.2),
  callbacks = list(
    callback_early_stopping(patience = as.integer(10), restore_best_weights = TRUE)
  )
)

model$evaluate(x_test, y_test)


# Predict log(skadkost)
log_preds_test <- model$predict(x_test)

# Transform back to original scale
preds_test <- exp(log_preds_test)
actual_test <- test_model$skadkost  # Already in original scale

# Compute metrics
mae_nn <- mean(abs(preds_test - actual_test))
mse_nn <- mean((preds_test - actual_test)^2)
rmse_nn <- sqrt(mse_nn)

#################################################################################
library(statmod)
library(glm2)
# Keep unscaled numeric predictors but ensure skadkost > 0
train_glm <- train_data %>%
  select(agarald, fordald, bonuskl, kon, zon, mcklass, duration, skadkost) %>%
  mutate(log_duration = log(duration))

test_glm <- test_data %>%
  select(agarald, fordald, bonuskl, kon, zon, mcklass, duration, skadkost) %>%
  mutate(log_duration = log(duration))

# Build design matrix for number of predictors
X <- model.matrix(
  skadkost ~ agarald + fordald + bonuskl + kon + zon + mcklass,
  data = train_glm
)

# Manually create starting values:  
# intercept = log(mean(skadkost/duration)), rest = zeros
start_intercept <- log(mean(train_glm$skadkost / train_glm$duration))
start_vals <- c(start_intercept, rep(0, ncol(X) - 1))

# Fit IG GLM using glm2 with manual start
glm_ig <- glm2(
  skadkost ~ agarald + fordald + bonuskl + kon + zon + mcklass + offset(log_duration),
  data = train_glm,
  family = inverse.gaussian(link = "log"),
  start = start_vals,
  control = glm.control(maxit = 1000, epsilon = 1e-8)
)

summary(glm_ig)

# Predict on test set
preds_ig <- predict(glm_ig, newdata = test_glm, type = "response")

# Evaluate metrics
mae_in <- mean(abs(preds_ig - test_glm$skadkost))
mse_in <- mean((preds_ig - test_glm$skadkost)^2)
rmse_in <- sqrt(mse_in)

#################################################################################
# GAM - Generalized Additive Model (Gamma family with log link)
library(mgcv)

# Use original training and testing data (not scaled, not one-hot)
train_gam <- train_data %>%
  mutate(log_duration = log(duration))

test_gam <- test_data %>%
  mutate(log_duration = log(duration))

# Fit GAM
gam_model <- gam(
  skadkost ~ s(agarald, k = 3) + s(fordald, k = 3) + s(bonuskl, k = 3) +
    kon + zon + mcklass +
    offset(log_duration),
  data = train_gam,
  family = Gamma(link = "log"),
  method = "REML"
)

summary(gam_model)

# Predict on test set
preds_gam <- predict(gam_model, newdata = test_gam, type = "response")

# Evaluate
mae_gam <- mean(abs(preds_gam - test_gam$skadkost))
mse_gam <- mean((preds_gam - test_gam$skadkost)^2)
rmse_gam <- sqrt(mse_gam)

#################################################################################
# GLM lognorm
# Copy the training data
train_model_log <- train_data

# Apply log transformation to the target (lognormal assumption)
train_model_log$skadkost <- log(train_model_log$skadkost)

# Fit Gaussian GLM on log scale (lognormal model)
glm_lognorm <- glm(
  skadkost ~ agarald + fordald + bonuskl + kon + zon + mcklass,
  data = train_model_log,
  family = gaussian(link = "identity"),   # Gaussian on log scale = lognormal
  offset = log(duration),
  control = glm.control(maxit = 1000, epsilon = 1e-8)
)

# Model summary
summary(glm_lognorm)

# Predict log-scale values on test set
test_data$log_pred <- predict(glm_lognorm, newdata = test_data, type = "response")

# Back-transform to original scale
test_data$predicted <- exp(test_data$log_pred)

# Evaluate performance
mae_lognorm <- mean(abs(test_data$predicted - test_data$skadkost))
mse_lognorm <- mean((test_data$predicted - test_data$skadkost)^2)
rmse_lognorm <- sqrt(mse_lognorm)

#################################################################################
# Random Forest
library(randomForest)
rf_model <- randomForest(skadkost ~ ., data = train_model, ntree = 500, mtry = 5)  # Tune mtry if needed
importance(rf_model)  # Feature importance

y_test <- as.numeric(y_test)

# Predict and evaluate
preds <- predict(rf_model, newdata = test_model)
str(preds)
mae_rf <- mean(abs(preds - y_test))
mse_rf <- mean((preds - y_test)^2)
rmse_rf <- sqrt(mse_rf)

#################################################################################
# GBM via XGboost
library(xgboost)
# Ensure pure numeric matrix and vector
x_train_xgb <- as.matrix(train_features)  # no data frames
y_train_xgb <- as.numeric(train_model$skadkost)

# Create DMatrix
dtrain <- xgb.DMatrix(data = x_train_xgb, label = y_train_xgb)
dtest <- xgb.DMatrix(data = as.matrix(x_test), label = as.numeric(y_test))
params <- list(objective = "reg:gamma",  # Gamma objective for skewed data
               eta = 0.01, max_depth = 6, nrounds = 1000)
xgb_model <- xgb.train(params, dtrain, nrounds = 1000, watchlist = list(eval = dtest), early_stopping_rounds = 50)

# Predict and evaluate
preds <- predict(xgb_model, dtest)
mae_gbm <- mean(abs(preds - y_test))
mse_gbm <- mean((preds - y_test)^2)
rmse_gbm <- sqrt(mse_gbm)
print(rmse_gbm)

#################################################################################
# Support Vector Regression
svr_model <- svm(skadkost ~ ., data = train_model, kernel = "radial", cost = 10, epsilon = 0.1)  # Tune cost/epsilon

# Predict and evaluate
preds <- predict(svr_model, newdata = test_model)
mae_svr <- mean(abs(preds - y_test))
mse_svr <- mean((preds - y_test)^2)
rmse_svr <- sqrt(mse_svr)

#################################################################################
# GLM + NN
library(caret)
library(caretEnsemble)

# Prepare train/test sets for caret
train_caret <- train_data %>%
  select(skadkost, agarald, fordald, bonuskl, kon, zon, mcklass, duration)

test_caret <- test_data %>%
  select(skadkost, agarald, fordald, bonuskl, kon, zon, mcklass, duration)

# Offset log(duration) manually by adding as a feature
train_caret$log_duration <- log(train_caret$duration)
test_caret$log_duration <- log(test_caret$duration)

# Remove duration (since log_duration replaces it)
train_caret <- train_caret %>% select(-duration)
test_caret <- test_caret %>% select(-duration)

options(nnet.MaxNWts = 5000)

# Define training control
set.seed(123)
train_ctrl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = "final",
  allowParallel = TRUE
)

models <- caretList(
  skadkost ~ .,
  data = train_caret,
  trControl = train_ctrl,
  methodList = c("glm"),
  tuneList = list(
    nnet = caretModelSpec(
      method = "nnet",
      tuneLength = 3,
      trace = FALSE,
      linout = TRUE,
      preProcess = c("center", "scale")  # <— THIS helps avoid NA metrics
    )
  )
)


# Build ensemble (stacked model)
ensemble_model <- caretEnsemble(
  models,
  metric = "RMSE",
  trControl = trainControl(
    method = "cv",
    number = 5,
    savePredictions = "final"
  )
)

# Predictions on test set
preds <- predict(ensemble_model, newdata = test_caret)

# If it's not numeric, coerce properly
if (is.data.frame(preds)) {
  preds <- preds[[1]]  # first column
}
preds <- as.numeric(preds)

# Actual values
actuals <- as.numeric(test_caret$skadkost)

# Metrics
mae_hybrid <- mean(abs(preds - actuals))
mse_hybrid <- mean((preds - actuals)^2)
rmse_hybrid <- sqrt(mse_hybrid)

#################################################################################
# GAM + NN
library(nnet)
# Step 1: Fit GAM
gam_model <- gam(
  skadkost ~ 
    s(agarald, k = 5) +       # fewer knots
    s(fordald, k = 5) +
    s(bonuskl, k = 5) +
    kon + zon + mcklass +
    offset(log(duration)),
  data = train_data,
  family = gaussian(link = "log")
)

# Step 2: Get GAM residuals
gam_preds <- predict(gam_model, newdata = train_data, type = "response")
residuals_gam <- train_data$skadkost - gam_preds

# Step 3: Train NN on residuals
set.seed(123)
nn_model <- nnet(
  x = model.matrix(~ agarald + fordald + bonuskl + kon + zon + mcklass, data = train_data),
  y = residuals_gam,
  size = 5,
  linout = TRUE,
  maxit = 500,
  trace = FALSE
)

# Step 4: Predictions (additive: GAM + NN residuals)
gam_test_preds <- as.numeric(predict(gam_model, newdata = test_data, type = "response"))

nn_test_preds <- as.numeric(predict(
  nn_model,
  newdata = model.matrix(~ agarald + fordald + bonuskl + kon + zon + mcklass, data = test_data)
))

# Combine predictions
hybrid_preds <- gam_test_preds + nn_test_preds


# Step 5: Metrics
mae_hybrid_2 <- mean(abs(hybrid_preds - test_data$skadkost))
mse_hybrid_2 <- mean((hybrid_preds - test_data$skadkost)^2)
rmse_hybrid_2 <- sqrt(mse_hybrid_2)

#################################################################################
# GLM + NN + GBM
# Step 2: Ensemble Stacking GLM + NN + GBM
set.seed(123)

# Train control for caret
train_ctrl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = "final",
  allowParallel = TRUE
)

# Prepare training data
train_caret <- train_data[, c("agarald", "fordald", "bonuskl", "kon", "zon", "mcklass", "skadkost")]

# Build base models
models <- caretList(
  skadkost ~ .,
  data = train_caret,
  trControl = train_ctrl,
  tuneList = list(
    glm = caretModelSpec(method = "glm"),
    nnet = caretModelSpec(
      method = "nnet",
      tuneLength = 3,
      trace = FALSE,
      linout = TRUE,
      preProcess = c("center", "scale")
    ),
    gbm = caretModelSpec(
      method = "gbm",
      tuneLength = 3,
      verbose = FALSE
    )
  )
)

# Stack models into ensemble
ensemble_model <- caretEnsemble(
  models,
  metric = "RMSE",
  trControl = trainControl(
    method = "cv",
    number = 5,
    savePredictions = "final"
  )
)

# Prepare test set
test_caret <- test_data[, c("agarald", "fordald", "bonuskl", "kon", "zon", "mcklass", "skadkost")]

# Predictions and evaluation
preds_ensemble <- unlist(predict(ensemble_model, newdata = test_caret))

mae_ens <- mean(abs(preds_ensemble - test_caret$skadkost))
mse_ens <- mean((preds_ensemble - test_caret$skadkost)^2)
rmse_ens <- sqrt(mse_ens)


results <- data.frame(
  Model = c("GAMMA GLM", "NN Gamma", "INV GAUSS GLM", "GAM", "GLM LOGNORMAL", "RANDOM FOREST","GBM", "SVR", "HYBRID GLM + NN", "HYBRID GAM + NN", "GLM + NN + GAM"),  # Add your models
  MAE = c(mae_glm, mae_nn, mae_in, mae_gam, mae_lognorm, mae_rf, mae_gbm, mae_svr, mae_hybrid, mae_hybrid_2, mae_ens),
  MSE = c(mse_glm, mse_nn, mse_in, mse_gam, mse_lognorm, mse_rf, mse_gbm, mse_svr, mse_hybrid, mse_hybrid_2, mse_ens),
  RMSE = c(rmse_glm, rmse_nn, rmse_in, rmse_gam, rmse_lognorm, rmse_rf, rmse_gbm, rmse_svr, rmse_hybrid, rmse_hybrid_2, rmse_ens)
)
print(results)