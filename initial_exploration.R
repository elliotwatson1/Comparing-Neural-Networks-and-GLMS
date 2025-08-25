# Load libraries
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

####################################################################################################################
############################################# BEGINNING OF EDA SECTION #############################################
####################################################################################################################
# LOADING DATA AND INITIAL VIEWS

data(dataOhlsson)
data = dataOhlsson

str(data)
summary(data)
head(data)

# INITIAL INSIGHTS
# Count missing values
sapply(data, function(x) sum(is.na(x))) # No missing values

# Summary of variable types and example anomalies
summary(data)

# Average claim cost by zone (bar plot)
ggplot(data, aes(x = as.factor(zon), y = skadkost)) +
  stat_summary(fun = mean, geom = "bar", fill = "steelblue") +
  labs(title = "Average Claim Cost by Zone", x = "Zone", y = "Avg skadkost")



# CODE FOR UNIVARIATE ANALYSIS
# Univariate analysis
# Histogram of numeric variables
numeric_vars <- c("agarald", "fordald", "bonuskl", "duration", "antskad", "skadkost")

data %>%
  select(all_of(numeric_vars)) %>%
  gather(variable, value) %>%
  ggplot(aes(x = value)) +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  geom_histogram(bins = 30, fill = "skyblue", color = "white") +
  theme_minimal() +
  labs(title = "Histograms of Numeric Variables")

# Create side-by-side histograms
# par(mfrow = c(1, 2))  # Set up a 1x2 plotting grid

data <- data[data$skadkost > 0, ]

# Histogram for skadkost
hist(data$skadkost, 
     main = "Histogram of skadkost",
     xlab = "skadkost",
     col = "lightblue",
     breaks = 100,
     border = "black",
     ylim = c(0,100))  # 👈 Proper format for xlim


# Histogram for antskad
#hist(data$antskad, 
#     main = "Histogram of antskad",
#     xlab = "antskad",
#     col = "lightgreen",
#     border = "black",
#     breaks = 100,
#     xlim = c(0,2))
     

# Reset the plotting layout
# par(mfrow = c(1, 1))

# Bar plots of categorical variables
data %>%
  select(kon, zon, mcklass) %>%
  gather(variable, value) %>%
  ggplot(aes(x = as.factor(value))) +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  geom_bar(fill = "salmon") +
  theme_minimal() +
  labs(title = "Bar Plots of Categorical Variables", x = "Category", y = "Count")

# CODE FOR BIVARIATE ANALYSIS
## Bivariate Analysis
# Correlation Matrix of Numeric Variables
data %>%
  select(all_of(numeric_vars)) %>%
  cor() %>%
  round(2) %>%
  reshape2::melt() %>%
  ggplot(aes(Var1, Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "white") +
  scale_fill_gradient2(midpoint = 0, low = "green", mid = "black", high = "orange", limits = c(-1, 1)) +
  theme_minimal() +
  labs(title = "Correlation Matrix of Numeric Variables")


# Zero inflation check
zero_freq <- mean(data$antskad == 0)
zero_cost <- mean(data$skadkost == 0)
cat("Percentage of zero claim frequency:", round(100 * zero_freq, 2), "%\n")
cat("Percentage of zero claim cost:", round(100 * zero_cost, 2), "%\n")

# Claim rate by age group
data %>%
  mutate(age_group = cut(agarald, breaks = seq(0, 100, 10), right = FALSE)) %>%
  group_by(age_group) %>%
  summarise(mean_claims = mean(antskad)) %>%
  ggplot(aes(x = age_group, y = mean_claims)) +
  geom_col(fill = "steelblue") +
  theme_minimal() +
  labs(title = "Average Claim Count by Driver Age Group", x = "Age Group", y = "Mean Claims")


# Average claim frequency and cost by Gender
p1_freq <- ggplot(data, aes(x = kon, y = antskad)) +
  stat_summary(fun = mean, geom = "bar", fill = "steelblue") +
  labs(title = "Average Claim Frequency by Gender", x = "Gender", y = "Mean Claims")

p1_cost <- ggplot(data, aes(x = kon, y = skadkost)) +
  stat_summary(fun = mean, geom = "bar", fill = "lightblue") +
  labs(title = "Average Claim Cost by Gender", x = "Gender", y = "Mean Cost")

grid_gender <- grid.arrange(p1_freq, p1_cost, ncol = 2)

# Average claim frequency and cost by Zone
p2_freq <- ggplot(data, aes(x = factor(zon), y = antskad)) +
  stat_summary(fun = mean, geom = "bar", fill = "darkorange") +
  labs(title = "Average Claim Frequency by Zone", x = "Zone", y = "Mean Claims")

p2_cost <- ggplot(data, aes(x = factor(zon), y = skadkost)) +
  stat_summary(fun = mean, geom = "bar", fill = "moccasin") +
  labs(title = "Average Claim Cost by Zone", x = "Zone", y = "Mean Cost")

grid_zone <- grid.arrange(p2_freq, p2_cost, ncol = 2)

# Average claim frequency and cost by Motorcycle Class
p3_freq <- ggplot(data, aes(x = factor(mcklass), y = antskad)) +
  stat_summary(fun = mean, geom = "bar", fill = "seagreen") +
  labs(title = "Average Claim Frequency by Motorcycle Class", x = "MC Class", y = "Mean Claims")

p3_cost <- ggplot(data, aes(x = factor(mcklass), y = skadkost)) +
  stat_summary(fun = mean, geom = "bar", fill = "mediumseagreen") +
  labs(title = "Average Claim Cost by Motorcycle Class", x = "MC Class", y = "Mean Cost")

grid_mcclass <- grid.arrange(p3_freq, p3_cost, ncol = 2)

zero_freq <- mean(data$antskad == 0)
zero_cost <- mean(data$skadkost == 0)

cat("Percentage of zero claim frequency:", round(100 * zero_freq, 2), "%\n")
cat("Percentage of zero claim cost:", round(100 * zero_cost, 2), "%\n")

# CODE FOR PCA
# PCA plot
numeric_data <- data[, c("agarald", "fordald", "bonuskl", "duration", "skadkost")]
pca_result <- prcomp(numeric_data, scale. = TRUE)
summary(pca_result)
biplot(pca_result, main = "PCA Biplot")

# CODE FOR MODEL FITTING
# Poisson fit for antskad
fit_pois <- fitdist(dataOhlsson$antskad, "pois")
summary(fit_pois)
plot(fit_pois)

# Gamma fit for skadkost > 0
skadkost_nonzero <- data$skadkost[data$skadkost > 0]
fit_gamma <- fitdist(skadkost_nonzero, "gamma")
summary(fit_gamma)
plot(fit_gamma)

# Some tests to see if the data is Gamma distributed
# Poisson fit for claim frequency
# Chi-square goodness-of-fit test
observed_counts <- table(data$antskad)
lambda_hat <- mean(data$antskad)
expected_counts <- dpois(as.numeric(names(observed_counts)), lambda = lambda_hat) * length(data$antskad)

# Combine tail categories if expected counts are too small
# (Chi-square test assumption: expected count >= 5 in each cell)
while(any(expected_counts < 5)) {
  # Merge last two categories
  last_idx <- length(expected_counts)
  expected_counts[last_idx-1] <- expected_counts[last_idx-1] + expected_counts[last_idx]
  observed_counts[last_idx-1] <- observed_counts[last_idx-1] + observed_counts[last_idx]
  expected_counts <- expected_counts[-last_idx]
  observed_counts <- observed_counts[-last_idx]
}

chisq_test <- chisq.test(observed_counts, p = expected_counts / sum(expected_counts))
chisq_test

# Gamma fit for positive claim costs
# Kolmogorov–Smirnov test
# Kolmogorov–Smirnov test for Gamma fit
ks_test_gamma <- ks.test(
  skadkost_nonzero,
  "pgamma",
  shape = fit_gamma$estimate["shape"],
  rate = fit_gamma$estimate["rate"]
)
ks_test_gamma

# GOF stats from fitdistrplus (includes Anderson-Darling, AIC, BIC, etc.)
gof_results <- gofstat(fit_gamma, fitnames = "Gamma")
gof_results





####################################################################################################################
################################# BEGINNING OF GLM SECTION #########################################################
####################################################################################################################

############################ FREQUENCY MODELLING ###################################################################

# Claim Frequency Modelling (Poisson GLM)

# Keep all policies with positive exposure (duration)
data <- data[data$duration > 0, ]

# Ensure categorical variables are factors
data$zon <- as.factor(data$zon)
data$kon <- as.factor(data$kon)
data$mcklass <- as.factor(data$mcklass)

# Check for non-finite values
sapply(data[, c("antskad", "agarald", "fordald", "bonuskl", "kon", "zon", "mcklass", "duration")],
       function(x) any(!is.finite(x)))

# Data split
set.seed(123)
train_index <- createDataPartition(data$antskad, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Verify offset log(duration)
summary(log(train_data$duration))
any(!is.finite(log(train_data$duration)))  # Should be FALSE

# Fit simple intercept-only model
simple_model <- glm(
  antskad ~ 1,
  family = poisson(link = "log"),
  offset = log(duration),
  data = train_data
)

# Fit Poisson GLM with offset
glm_pois <- glm2(
  antskad ~ agarald + fordald + bonuskl + kon + zon + mcklass,
  family = poisson(link = "log"),
  offset = log(duration),
  data = train_data,
  start = c(coef(simple_model), rep(0, 16)),  # optional: create good starting values
  control = glm.control(maxit = 1000, epsilon = 1e-8)
)

# Summary of the model
summary(glm_pois)

# Predict on test set
test_data$predicted <- predict(glm_pois, newdata = test_data, type = "response")

# Evaluate performance on test set
mse <- mean((test_data$antskad - test_data$predicted)^2)
rmse <- sqrt(mse)
mae <- mean(abs(test_data$antskad - test_data$predicted))
cat("Test MSE:", round(mse, 4), "\n")
cat("Test RMSE:", round(rmse, 4), "\n")
cat("Test MAE:", round(mae, 4), "\n")

# Deviance
cat("Model Deviance:", glm_pois$deviance, "\n")

y <- test_data$skadkost
mu_hat <- test_data$predicted

test_deviance <- 2 * sum(((y - mu_hat) / mu_hat) - log(y / mu_hat))
test_deviance

# Psuedo-R^2
psr2 <- 1 - (glm_pois$deviance / simple_model$deviance)
cat("Pseudo-R^2:", round(psr2, 4), "\n")

# Plot predicted vs actual
ggplot(test_data, aes(x = antskad, y = predicted)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Poisson GLM: Predicted vs Actual Number of Claims",
       x = "Actual Claims (antskad)", y = "Predicted Claims") +
  theme_minimal()

# Residual diagnostics
fitted_values <- fitted(glm_pois)
deviance_resid <- residuals(glm_pois, type = "deviance")

plot(fitted_values, deviance_resid, pch = 20, col = "#1E90FF",
     main = "Deviance Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Deviance Residuals")
abline(h = 0, col = "red", lty = 2)

# Standardised residuals
std_resid <- rstandard(glm_pois)
plot(fitted_values, std_resid, pch = 20, col = "#1E90FF",
     main = "Standardized Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Standardized Residuals")
abline(h = 0, col = "red", lty = 2)

# Leverage plot
leverage <- hatvalues(glm_pois)
plot(leverage, pch = 20, col = "#1E90FF",
     main = "Leverage Values",
     ylab = "Leverage")
abline(h = 2 * mean(leverage), col = "red", lty = 2)

# Cook's Distance
cooksd <- cooks.distance(glm_pois)
plot(cooksd, pch = 20, col = "#1E90FF",
     main = "Cook’s Distance",
     ylab = "Cook’s Distance")
abline(h = 4 / nrow(train_data), col = "red", lty = 2)

# Overdispersion check
dispersion_ratio <- summary(glm_pois)$deviance / summary(glm_pois)$df.residual
p_val_overdisp <- pchisq(summary(glm_pois)$deviance, df = df.residual(glm_pois), lower.tail = FALSE)

cat("Dispersion ratio:", round(dispersion_ratio, 4), "\n")
cat("Overdispersion test p-value:", round(p_val_overdisp, 4), "\n")

# Influential observations
influential <- which(cooksd > 4 / nrow(train_data))
cat("Influential observations:", length(influential), "\n")

##################################################### SEVERITY MODELLING #######################

# Filter out zero claim costs and zero durations
data <- data[data$skadkost > 0 & data$duration > 0, ]


# Ensure categorical variables are factors
data$zon <- as.factor(data$zon)
data$kon <- as.factor(data$kon)
data$mcklass <- as.factor(data$mcklass)

# Check for non-finite values
sapply(data[, c("skadkost", "agarald", "fordald", "bonuskl", "kon", "zon", "mcklass", "duration")],
       function(x) any(!is.finite(x)))

# Data split
set.seed(123) 
train_index <- createDataPartition(data$skadkost, p = 0.8, list = FALSE) # 80:20 split
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Verify duration in training data
summary(log(train_data$duration))
any(!is.finite(log(train_data$duration))) # Should be FALSE

# Get the design matrix
X <- model.matrix(
  ~ agarald + fordald + bonuskl + kon + zon + mcklass,
  data = train_data
)

summary(X)
ncol(X) 
any(!is.finite(X)) # Should be FALSE
qr(X)$rank == ncol(X) # Should be TRUE for full rank

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
mse <- mean((test_data$skadkost - test_data$predicted)^2)
rmse <- sqrt(mse)
mae <- mean(abs(test_data$skadkost - test_data$predicted))
cat("Test MSE:", round(mse, 4), "\n")
cat("Test RMSE:", round(rmse, 4), "\n")
cat("Test MAE:", round(mae, 4), "\n")

# Plot predicted vs actual on log scale
ggplot(test_data, aes(x = skadkost, y = predicted)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  scale_x_log10() + scale_y_log10() +
  labs(title = "Gamma GLM: Predicted vs Actual on Test Set (Log Scale)",
       x = "Actual Claim Cost", y = "Predicted Claim Cost") +
  theme_minimal()

# Deviance
glm_gamma$deviance

# Psuedo-R^2
psr2 = 1 - (glm_gamma$deviance / simple_model$deviance)
psr2

y <- test_data$skadkost
mu_hat <- test_data$predicted

test_deviance <- 2 * sum(((y - mu_hat) / mu_hat) - log(y / mu_hat))
test_deviance

# Extract fitted values and deviance residuals
fitted_values <- fitted(glm_gamma)
deviance_resid <- residuals(glm_gamma, type = "deviance")

# Plot - Residual analysis
plot(fitted_values, deviance_resid, pch = 20, col = "#1E90FF",
     main = "Deviance Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Deviance Residuals")
abline(h = 0, col = "red", lty = 2)

# Plot - Standardised residuals
std_resid <- rstandard(glm_gamma)
plot(fitted_values, std_resid, pch = 20, col = "#1E90FF",
     main = "Standardized Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Standardized Residuals")
abline(h = 0, col = "red", lty = 2)

# Plot - leverage plot
leverage <- hatvalues(glm_gamma)
plot(leverage, pch = 20, col = "#1E90FF",
     main = "Leverage Values",
     ylab = "Leverage")
abline(h = 2 * mean(leverage), col = "red", lty = 2) # Rule of thumb threshold

# Plot - Cook's Distance
cooksd <- cooks.distance(glm_gamma)
plot(cooksd, pch = 20, col = "#1E90FF",
     main = "Cook’s Distance",
     ylab = "Cook’s Distance")
abline(h = 4/nrow(train_data), col = "red", lty = 2) # Threshold

# Check for over dispersion
pchisq(summary(glm_gamma)$deviance, df = df.residual(glm_gamma), lower.tail = FALSE)

# Influential points
influential <- which(cooksd > 4/nrow(train_data))
print(influential) # Indices of influential points

####################################################################################################################
# OUT OF SAMPLE
####################################################################################################################
# Get predictions on test set
mu_test <- predict(glm_gamma, newdata = test_data, type = "response")

# Extract observed values
y_test <- test_data$skadkost

# Calculate Gamma deviance (unit weights, log link)
# Formula: 2 * sum((y - mu)/mu - log(y/mu))
test_deviance <- 2 * sum((y_test - mu_test) / mu_test - log(y_test / mu_test))

cat("Out-of-sample Deviance:", round(test_deviance, 4), "\n")

# Calculate deviance residuals for test data
dev_resid <- sign(y_test - mu_test) * sqrt(
  2 * ((y_test - mu_test) / mu_test - log(y_test / mu_test))
)

test_data$dev_resid <- dev_resid

# Plot deviance residuals vs predicted values
ggplot(test_data, aes(x = mu_test, y = dev_resid)) +
  geom_point(alpha = 0.4) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Out-of-Sample Deviance Residuals",
    x = "Predicted Claim Cost (μ)",
    y = "Deviance Residual"
  ) +
  theme_minimal()

library(dplyr)

deviance_df <- test_data %>%
  mutate(abs_dev = (y_test - mu_test) / mu_test - log(y_test / mu_test)) %>%
  mutate(pred_bin = ntile(mu_test, 10)) %>%  # divide into 10 bins by prediction
  group_by(pred_bin) %>%
  summarise(total_dev = 2 * sum(abs_dev), .groups = "drop")

ggplot(deviance_df, aes(x = pred_bin, y = total_dev)) +
  geom_col(fill = "steelblue") +
  labs(
    title = "Out-of-Sample Deviance by Prediction Decile",
    x = "Prediction Decile",
    y = "Total Deviance"
  ) +
  theme_minimal()

# Get predictions on test set
mu_test <- predict(glm_gamma, newdata = test_data, type = "response")
y_test <- test_data$skadkost

# Compute out-of-sample deviance for Gamma GLM (log link)
test_deviance <- 2 * sum((y_test - mu_test) / mu_test - log(y_test / mu_test))

# Add predictions to test_data for plotting
test_data$predicted <- mu_test

# Plot Actual vs Predicted (log scale) + deviance overlay
library(ggplot2)

ggplot(test_data, aes(x = skadkost, y = predicted)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  scale_x_log10() + scale_y_log10() +
  labs(
    title = "Gamma GLM: Predicted vs Actual on Test Set (Log Scale)",
    subtitle = paste("Out-of-sample Deviance:", round(test_deviance, 2)),
    x = "Actual Claim Cost",
    y = "Predicted Claim Cost"
  ) +
  theme_minimal()



####################################################################################################################
############################################### END OF GLM SECTION #################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
############################################ BEGGINING OF NN SECTION ###############################################
####################################################################################################################
library(insuranceData)

data(dataOhlsson)
data = dataOhlsson

str(data)
summary(data)
head(data)

data <- data[data$skadkost > 0 & data$duration > 0, ]

# Data split
set.seed(123) 
train_index <- createDataPartition(data$skadkost, p = 0.8, list = FALSE) # 80:20 split
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

############### IGNORE - UNLESS RUNNING FROM COMPLETELY FRESH ##################
installed.packages("keras")
library(reticulate)
install_tensorflow()
library(keras)
library(tensorflow)
use_condaenv("r-tensorflow", required = TRUE)




py_install(c("pydot", "graphviz"))

tf$config$list_physical_devices()
################################################################################


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

print(dim(x_train))
print(dim(y_train))
print(anyNA(x_train))
print(anyNA(y_train))
print(dim(x_test))
print(dim(y_test))
print(anyNA(x_test))
print(anyNA(y_test))

ncol(x_train)

x_train <- np_array(x_train)
y_train <- np_array(y_train)

x_test <- np_array(x_test)
y_test <- np_array(y_test)

############################# MSE MODEL ########################################

model_mse <- keras_model_sequential() 
model_mse$add(layer_dense(units = 64, activation = "relu", input_shape = c(ncol(x_train))))
model_mse$add(layer_dropout(rate = 0.3))
model_mse$add(layer_dense(units = 32, activation = "relu"))
model_mse$add(layer_dropout(rate = 0.3))
model_mse$add(layer_dense(units = 1))

model_mse$compile(
  loss = "mse",
  optimizer = optimizer_adam(),
  metrics = list("mean_squared_error", "mean_absolute_error")
)

history <- model_mse$fit(
  x = x_train,
  y = y_train,
  epochs = as.integer(100),
  batch_size = as.integer(256),
  validation_split = as.integer(0.2),
  callbacks = list(
    callback_early_stopping(patience = as.integer(10), restore_best_weights = TRUE)
  )
)

length(data$skadkost)

model_mse$evaluate(x_test, y_test)

# Predict and back-transform
log_preds_test <- model_mse$predict(x_test)
preds_test <- exp(log_preds_test)

# Back-transform target too for consistency
actual_test <- test_model$skadkost

# Performance metrics
mae <- mean(abs(preds_test - actual_test))
rmse <- sqrt(mean((preds_test - actual_test)^2))
mse <- mean((preds_test - actual_test)^2)

cat("MAE:", mae, "\n")
cat("RMSE:", rmse, "\n")
cat("MSE:", mse, "\n")

if (any(actual_test <= 0) || any(preds_test <= 0)) {
  stop("Gamma deviance is undefined for zero or negative values.")
}

# Compute Gamma deviance for neural network predictions
gamma_deviance_nn <- 2 * sum(((actual_test - preds_test) / preds_test) - log(actual_test / preds_test))
cat("Gamma Deviance (Neural Network):", gamma_deviance_nn, "\n")

# Visualise
plot_df <- data.frame(
  Actual = actual_test,
  Predicted = preds_test
)

ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(colour = "red") +
  labs(title = "Actual vs Predicted", x = "Actual skadkost", y = "Predicted") +
  theme_minimal()

############################# GAMMA MODEL ########################################
gamma_loss <- function(y_true, y_pred){
  K <- backend()
  y_true <- K$clip(y_true, K$epsilon(), K$cast_to_floatx(1e10))
  y_pred <- K$clip(y_pred, K$epsilon(), K$cast_to_floatx(1e10))
  loss <- 2 * ((y_true - y_pred) / y_pred - K$log(y_true / y_pred))
  K$mean(loss, axis = as.integer(-1))
}

model_gamma <- keras_model_sequential() 
model_gamma$add(layer_dense(units = 64, activation = "relu", input_shape = c(ncol(x_train))))
model_gamma$add(layer_dropout(rate = 0.3))
model_gamma$add(layer_dense(units = 32, activation = "relu"))
model_gamma$add(layer_dropout(rate = 0.3))
model_gamma$add(layer_dense(units = 1))

model_gamma$compile(
  loss = gamma_loss,
  optimizer = optimizer_adam(),
  metrics = list("mean_squared_error", "mean_absolute_error")
)

history_gamma <- model_gamma$fit(
  x = x_train,
  y = y_train,
  epochs = as.integer(100),
  batch_size = as.integer(256),
  validation_split = as.integer(0.2),
  callbacks = list(
    callback_early_stopping(patience = as.integer(10), restore_best_weights = TRUE)
  )
)

preds_gamma <- model_gamma$predict(x_test)

log_preds_mse <- model_mse$predict(x_test)
preds_mse <- exp(log_preds_mse)  # back-transform

compare_metrics <- function(preds, actual, label){
  mae <- mean(abs(preds - actual))
  rmse <- sqrt(mean((preds - actual)^2))
  mse <- mean((preds - actual)^2)
  
  cat("\nPerformance for", label, "model:\n")
  cat("MAE:", round(mae, 2), "\n")
  cat("RMSE:", round(rmse, 2), "\n")
  cat("MSE:", round(mse, 2), "\n")
}

compare_metrics(preds_mse, actual, "MSE")
compare_metrics(preds_gamma, actual, "Gamma")

###################### HYPERPARAMETER TUNING ##################################################

library(insuranceData)

data(dataOhlsson)
data = dataOhlsson

str(data)
summary(data)
head(data)

data <- data[data$skadkost > 0 & data$duration > 0, ]

# Data split
set.seed(123)
train_index <- createDataPartition(data$skadkost, p = 0.8, list = FALSE) # 80:20 split
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

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

# Normalise features as before
train_features <- train_model %>% select(-skadkost) %>% mutate_all(normalise_train)
test_features <- test_model %>% select(-skadkost) %>% mutate(across(everything(), ~ normalise_test(.x, cur_column())))

# Targets
train_targets <- train_model$skadkost
test_targets <- test_model$skadkost

# Assign x_train, y_train, x_test, y_test
x_train <- as.matrix(train_features)
y_train <- as.numeric(train_targets)

x_test <- as.matrix(test_features)
y_test <- as.numeric(test_targets)

# Check
print(dim(x_train))
print(length(y_train))
print(dim(x_test))
print(length(y_test))
print(anyNA(x_train))
print(anyNA(y_train))
print(anyNA(x_test))
print(anyNA(y_test))

library(tfruns)
runs <- tuning_run("train_model.R", 
                   flags = list(
                     units1 = c(64, 128,256),
                     units2 = c(32, 64, 128),
                     units3 = c(16, 32, 64),
                     dropout1 = c(0.2, 0.3),
                     activation = c("relu", "tanh", "elu"),
                     final_activation = "relu",  # Keep final output positive
                     optimizer = c("adam", "rmsprop", "sgd"),
                     num_layers = c(1, 2, 3),
                     batch_size = c(128, 256),
                     lr = c(0.001, 0.0005)
                   ))

#################################################################################
# NOW USING THE BEST VALUES FOUND IN LONG RUN
#################################################################################
library(insuranceData)
library(dplyr)

data(dataOhlsson)
data = dataOhlsson

str(data)
summary(data)
head(data)

data <- data[data$skadkost > 0 & data$duration > 0, ]

# Data split
set.seed(123) 
train_index <- createDataPartition(data$skadkost, p = 0.8, list = FALSE) # 80:20 split
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

installed.packages("keras")
library(reticulate)
use_condaenv("r-tensorflow", required = TRUE)
library(keras)
library(tensorflow)

vars <- c("skadkost", "agarald", "kon", "zon", "mcklass", "fordald", "bonuskl")

train_model <- train_data[, vars]
test_model  <- test_data[, vars]

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
train_features <- train_model %>% dplyr::select(-skadkost) %>% dplyr::mutate_all(normalise_train)

# Capture min/max for each variable
mins <- sapply(train_model %>% dplyr::select(-skadkost), min)
maxs <- sapply(train_model %>% dplyr::select(-skadkost), max)

# Apply same scaling to test set
normalise_test <- function(x, varname) (x - mins[varname]) / (maxs[varname] - mins[varname])
test_features <- test_model %>%
  dplyr::select(-skadkost) %>%
  mutate(across(everything(), ~ normalise_test(.x, cur_column())))

x_train <- as.matrix(train_features)
y_train <- log(train_model$skadkost / train_data$duration)
y_train_raw <- train_model$skadkost

x_test <- as.matrix(test_features)
y_test  <- log(test_model$skadkost / test_data$duration)
y_test_raw <- test_model$skadkost

library(reticulate)

np <- import("numpy")
# Convert to tensors
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
log_preds_test <- model$predict(x_test)        # model output
preds_test <- exp(log_preds_test) 

actual_test <- test_model$skadkost / test_data$duration # account for exposure

mae <- mean(abs(preds_test - actual_test))
mse <- mean((preds_test - actual_test)^2)
rmse <- sqrt(mse)

cat("MAE:", round(mae, 2), "\n")
cat("MSE:", round(mse, 2), "\n")
cat("RMSE:", round(rmse, 2), "\n")

plot_df <- data.frame(
  Actual = actual_test,
  Predicted = preds_test
)

ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Actual vs Predicted Claim Costs",
    x = "Actual skadkost",
    y = "Predicted skadkost"
  ) +
  theme_minimal() 

# Calculate R-squared manually
# Actual and predicted
y <- actual_test
mu <- preds_test

# Null model: predict mean of actual values
mu_null <- mean(y)

# Log-likelihood under Gamma (approx, log-link)
loglik_model <- sum(dgamma(y, shape = 1, scale = mu, log = TRUE))
loglik_null <- sum(dgamma(y, shape = 1, scale = mu_null, log = TRUE))

# McFadden Pseudo R²
pseudo_r2 <- 1 - (loglik_model / loglik_null)

cat("McFadden's Pseudo R²:", round(pseudo_r2, 4), "\n")

# Create a single data frame with all metrics
df <- data.frame(
  epoch = 1:length(history$history$loss),
  loss = unlist(history$history$loss),
  mae = unlist(history$history$mean_absolute_error),
  mse = unlist(history$history$mean_squared_error)
)

# Optional: remove first epoch
df <- df[-1, ]  # Exclude first row if needed

# Reshape to long format
df_long <- df %>%
  pivot_longer(cols = c("loss", "mae", "mse"),
               names_to = "metric",
               values_to = "value")

# Plot all three metrics
ggplot(df_long, aes(x = epoch, y = value, color = metric)) +
  geom_line(size = 1) +
  labs(
    title = "Training Metrics Over Epochs",
    x = "Epoch",
    y = "Metric Value",
    color = "Metric"
  ) +
  theme_minimal()


#################################################################################
# NOW USING THE WORST VALUES FOUND IN LONG RUN
#################################################################################
library(insuranceData)

data(dataOhlsson)
data = dataOhlsson

str(data)
summary(data)
head(data)

data <- data[data$skadkost > 0 & data$duration > 0, ]

# Data split
set.seed(123)
train_index <- createDataPartition(data$skadkost, p = 0.8, list = FALSE) # 80:20 split
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

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

library(reticulate)

np <- import("numpy")
# Convert to tensors
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
# Define model
model <- keras_model_sequential()
model$add(layer_dense(units = 256, activation = "elu", input_shape = c(ncol(x_train))))
model$add(layer_dropout(rate = 0.2))
model$add(layer_dense(units = 128, activation = "elu"))
model$add(layer_dense(units = 64, activation = "elu"))
model$add(layer_dense(units = 1, activation = "relu"))  # Final activation

# Compile with SGD optimizer
model$compile(
  loss = gamma_loss,  # Make sure gamma_loss is defined correctly
  optimizer = optimizer_sgd(learning_rate = 0.001),
  metrics = list("mean_squared_error", "mean_absolute_error")
)

# Train the model
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
log_preds_test <- model$predict(x_test)        # model output
preds_test <- exp(log_preds_test) 

actual_test <- test_model$skadkost / test_data$duration # account for exposure

# Compute metrics
mae <- mean(abs(preds_test - actual_test))
mse <- mean((preds_test - actual_test)^2)
rmse <- sqrt(mse)

cat("MAE:", round(mae, 2), "\n")
cat("MSE:", round(mse, 2), "\n")
cat("RMSE:", round(rmse, 2), "\n")

plot_df <- data.frame(
  Actual = actual_test,
  Predicted = preds_test
)

ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Actual vs Predicted Claim Costs",
    x = "Actual skadkost",
    y = "Predicted skadkost"
  ) +
  theme_minimal()

# Calculate R-squared manually
# Actual and predicted
y <- actual_test
mu <- preds_test

# Null model: predict mean of actual values
mu_null <- mean(y)

# Log-likelihood under Gamma (approx, log-link)
loglik_model <- sum(dgamma(y, shape = 1, scale = mu, log = TRUE))
loglik_null <- sum(dgamma(y, shape = 1, scale = mu_null, log = TRUE))

# McFadden Pseudo R²
pseudo_r2 <- 1 - (loglik_model / loglik_null)

cat("McFadden's Pseudo R²:", round(pseudo_r2, 4), "\n")

