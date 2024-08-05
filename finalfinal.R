# Load necessary libraries
library(ggplot2)
library(ggplotify)
library(gridExtra)
library(knitr)
library(patchwork)
library(tidyverse)
library(dplyr)
library(lubridate)
library(tidyr)
library(Metrics)
library(caret)




setwd("/Users/hesun/Desktop")

getwd()
# Read in the dataset
tennis_data <- read.csv("combined_tennis_data_2010_2019.csv")
tennis_data$Date <- as.Date(tennis_data$Date, format = "%Y-%m-%d")

# Determine if the higher-ranked player won
tennis_data$higher_rank_won <- tennis_data$WRank < tennis_data$LRank

# Calculate points for higher and lower-ranked players
tennis_data <- tennis_data %>%
  mutate(
    higher_rank_points = ifelse(higher_rank_won, WPts, LPts),
    lower_rank_points = ifelse(higher_rank_won, LPts, WPts),
    point_difference = higher_rank_points - lower_rank_points
  )

# Split the dataset into training, validation, and testing sets based on year
train_set <- tennis_data %>% filter(year(Date) < 2018)
validation_set <- tennis_data %>% filter(year(Date) >= 2018 & year(Date) < 2019)
test_set <- tennis_data %>% filter(year(Date) == 2019)

# Remove rows with missing values in the key columns
train_set <- train_set %>% drop_na()
validation_set <- validation_set %>% drop_na()
test_set <- test_set %>% drop_na()

# Function to calculate normalized and consensus probabilities
calculate_probabilities <- function(df) {
  df <- df %>%
    mutate(
      p1_b365 = B365L / (B365W + B365L),
      p2_b365 = B365W / (B365W + B365L),
      p1_ps = PSL / (PSW + PSL),
      p2_ps = PSW / (PSW + PSL)
    ) %>%
    mutate(
      consensus_p1 = (p1_b365 + p1_ps) / 2,
      consensus_p2 = (p2_b365 + p2_ps) / 2,
      higher_rank_prob = ifelse(WRank < LRank, consensus_p1, consensus_p2),
      lower_rank_prob = ifelse(WRank < LRank, consensus_p2, consensus_p1),
      predicted_winner = ifelse(higher_rank_prob > lower_rank_prob, "W", "L")
    )
  return(df)
}

# Apply the probability calculation function to the datasets
train_set <- calculate_probabilities(train_set)
validation_set <- calculate_probabilities(validation_set)
test_set <- calculate_probabilities(test_set)

# Function to evaluate the model performance
evaluate_performance <- function(df) {
  df <- df %>%
    mutate(
      actual_winner = ifelse(WRank < LRank, 1, 0)
    )
  
  correct_preds <- sum(df$predicted_winner == ifelse(df$actual_winner == 1, "W", "L"))
  total_preds <- nrow(df)
  accuracy <- correct_preds / total_preds
  
  log_loss <- -mean(df$actual_winner * log(df$higher_rank_prob) + 
                      (1 - df$actual_winner) * log(1 - df$higher_rank_prob), na.rm = TRUE)
  
  calibration <- sum(df$higher_rank_prob, na.rm = TRUE) / sum(df$actual_winner, na.rm = TRUE)
  
  list(
    accuracy = accuracy,
    log_loss = log_loss,
    calibration = calibration
  )
}

# Evaluate the model on the training, validation, and test sets
train_perf <- evaluate_performance(train_set)
validation_perf <- evaluate_performance(validation_set)
test_perf <- evaluate_performance(test_set)

# Compile performance metrics
performance_metrics <- data.frame(
  dataset = c("Training", "Validation", "Testing"),
  accuracy = c(train_perf$accuracy, validation_perf$accuracy, test_perf$accuracy),
  log_loss = c(train_perf$log_loss, validation_perf$log_loss, test_perf$log_loss),
  calibration = c(train_perf$calibration, validation_perf$calibration, test_perf$calibration)
)

# Print performance metrics
print(kable(performance_metrics, caption = "Model Performance Metrics"))


library(kableExtra)

# Compile performance metrics
performance_metrics <- data.frame(
  dataset = c("Training", "Validation", "Testing"),
  accuracy = c(train_perf$accuracy, validation_perf$accuracy, test_perf$accuracy),
  log_loss = c(train_perf$log_loss, validation_perf$log_loss, test_perf$log_loss),
  calibration = c(train_perf$calibration, validation_perf$calibration, test_perf$calibration)
)

# Print performance metrics
kable(performance_metrics, caption = "Model Performance Metrics for BCM") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                full_width = F, 
                position = "center")



















# Naive model using 2018 data
train_data_naive <- validation_set %>% filter(year(Date) == 2018)

win_prob_2018 <- mean(train_data_naive$higher_rank_won)

preds_2019 <- rep(win_prob_2018, nrow(test_set))
actual_outcomes_2019 <- test_set$higher_rank_won
naive_accuracy <- mean((preds_2019 > 0.5) == actual_outcomes_2019)

calibration_naive <- sum(preds_2019) / sum(actual_outcomes_2019)

log_loss_naive <- -mean(actual_outcomes_2019 * log(win_prob_2018) + 
                          (1 - actual_outcomes_2019) * log(1 - win_prob_2018), na.rm = TRUE)

naive_metrics <- data.frame(
  dataset = "Naive Testing",
  accuracy = naive_accuracy,
  log_loss = log_loss_naive,
  calibration = calibration_naive
)

# Append naive model metrics
performance_metrics <- rbind(performance_metrics, naive_metrics)



# Naive model metrics
naive_metrics <- data.frame(
  dataset = "Naive Testing",
  accuracy = naive_accuracy,
  log_loss = log_loss_naive,
  calibration = calibration_naive
)



# Print Naive model performance metrics
print(kable(naive_metrics, caption = "Naive Model Performance Metrics for naive") %>%
        kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                      full_width = F, 
                      position = "center"))


















# Logistic regression model
logistic_model <- glm(
  higher_rank_won ~ point_difference + 0,
  data = train_set,
  family = binomial(link = 'logit')
)

# Predict probabilities on the test set
test_probs_logistic <- predict(logistic_model, newdata = test_set, type = "response")
test_preds_logistic <- ifelse(test_probs_logistic > 0.5, 1, 0)

accuracy_logistic <- mean(test_preds_logistic == test_set$higher_rank_won)

log_loss_logistic <- -mean(test_set$higher_rank_won * log(test_probs_logistic) + 
                             (1 - test_set$higher_rank_won) * log(1 - test_probs_logistic), na.rm = TRUE)

calibration_logistic <- sum(test_probs_logistic) / sum(test_set$higher_rank_won)

logistic_metrics <- data.frame(
  dataset = "Logistic Testing",
  accuracy = accuracy_logistic,
  log_loss = log_loss_logistic,
  calibration = calibration_logistic
)

# Append logistic model metrics
performance_metrics <- rbind(performance_metrics, logistic_metrics)

# Print updated performance metrics
print(performance_metrics)




# Logistic model metrics
logistic_metrics <- data.frame(
  dataset = "Logistic Testing",
  accuracy = accuracy_logistic,
  log_loss = log_loss_logistic,
  calibration = calibration_logistic
)



# Print Logistic model performance metrics
print(kable(logistic_metrics, caption = "Logistic Model Performance Metrics for logistic model") %>%
        kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                      full_width = F, 
                      position = "center"))









# Function to calculate expected Elo score
expected_elo_score <- function(rating_a, rating_b) {
  1 / (1 + 10^((rating_b - rating_a) / 400))
}

# Function to update Elo ratings
update_elo_ratings <- function(winner_rating, loser_rating, k_factor = 25) {
  expected_winner <- expected_elo_score(winner_rating, loser_rating)
  new_winner_rating <- winner_rating + k_factor * (1 - expected_winner)
  new_loser_rating <- loser_rating + k_factor * (0 - (1 - expected_winner))
  return(c(new_winner_rating, new_loser_rating))
}

# Initialize Elo ratings for players
all_players <- unique(c(train_set$Winner, train_set$Loser, validation_set$Winner, validation_set$Loser, test_set$Winner, test_set$Loser))
elo_ratings <- setNames(rep(1500, length(all_players)), all_players)

# Update Elo ratings based on training data
for (i in 1:nrow(train_set)) {
  winner <- train_set$Winner[i]
  loser <- train_set$Loser[i]
  winner_rating <- elo_ratings[as.character(winner)]
  loser_rating <- elo_ratings[as.character(loser)]
  updated_ratings <- update_elo_ratings(winner_rating, loser_rating)
  elo_ratings[as.character(winner)] <- updated_ratings[1]
  elo_ratings[as.character(loser)] <- updated_ratings[2]
}


# Calculate expected probabilities using Elo ratings
calculate_elo_probs <- function(data, ratings) {
  probs <- numeric(nrow(data))
  for (i in 1:nrow(data)) {
    winner <- data$Winner[i]
    loser <- data$Loser[i]
    winner_rating <- ratings[as.character(winner)]
    loser_rating <- ratings[as.character(loser)]
    prob <- expected_elo_score(winner_rating, loser_rating)
    probs[i] <- ifelse(data$higher_rank_won[i], prob, 1 - prob)
  }
  return(probs)
}

# Calculate probabilities for train, validation, and test sets
train_probs_elo <- calculate_elo_probs(train_set, elo_ratings)
validation_probs_elo <- calculate_elo_probs(validation_set, elo_ratings)
test_probs_elo <- calculate_elo_probs(test_set, elo_ratings)

# Calculate metrics for the Elo model
calculate_metrics <- function(probs, actuals) {
  preds <- ifelse(probs > 0.5, 1, 0)
  accuracy <- mean(preds == actuals)
  log_loss <- -mean(actuals * log(probs) + (1 - actuals) * log(1 - probs), na.rm = TRUE)
  calibration <- sum(probs) / sum(actuals)
  return(list(accuracy = accuracy, log_loss = log_loss, calibration = calibration))
}

# Evaluate Elo model on train, validation, and test sets
elo_train_metrics <- calculate_metrics(train_probs_elo, train_set$higher_rank_won)
elo_validation_metrics <- calculate_metrics(validation_probs_elo, validation_set$higher_rank_won)
elo_test_metrics <- calculate_metrics(test_probs_elo, test_set$higher_rank_won)

# Compile Elo model metrics
elo_model_metrics <- data.frame(
  dataset = c("ELO Training", "ELO Validation", "ELO Testing"),
  accuracy = c(elo_train_metrics$accuracy, elo_validation_metrics$accuracy, elo_test_metrics$accuracy),
  log_loss = c(elo_train_metrics$log_loss, elo_validation_metrics$log_loss, elo_test_metrics$log_loss),
  calibration = c(elo_train_metrics$calibration, elo_validation_metrics$calibration, elo_test_metrics$calibration)
)

# Append Elo model metrics
performance_metrics <- rbind(performance_metrics, elo_model_metrics)

# Print updated performance metrics
print(performance_metrics)












library(kableExtra)

# Function to calculate expected Elo score
expected_elo_score <- function(rating_a, rating_b) {
  1 / (1 + 10^((rating_b - rating_a) / 400))
}

# Function to update Elo ratings
update_elo_ratings <- function(winner_rating, loser_rating, k_factor = 25) {
  expected_winner <- expected_elo_score(winner_rating, loser_rating)
  new_winner_rating <- winner_rating + k_factor * (1 - expected_winner)
  new_loser_rating <- loser_rating + k_factor * (0 - (1 - expected_winner))
  return(c(new_winner_rating, new_loser_rating))
}

# Initialize Elo ratings for players
all_players <- unique(c(train_set$Winner, train_set$Loser, validation_set$Winner, validation_set$Loser, test_set$Winner, test_set$Loser))
elo_ratings <- setNames(rep(1500, length(all_players)), all_players)

# Update Elo ratings based on training data
for (i in 1:nrow(train_set)) {
  winner <- train_set$Winner[i]
  loser <- train_set$Loser[i]
  winner_rating <- elo_ratings[as.character(winner)]
  loser_rating <- elo_ratings[as.character(loser)]
  updated_ratings <- update_elo_ratings(winner_rating, loser_rating)
  elo_ratings[as.character(winner)] <- updated_ratings[1]
  elo_ratings[as.character(loser)] <- updated_ratings[2]
}

# Calculate expected probabilities using Elo ratings
calculate_elo_probs <- function(data, ratings) {
  probs <- numeric(nrow(data))
  for (i in 1:nrow(data)) {
    winner <- data$Winner[i]
    loser <- data$Loser[i]
    winner_rating <- ratings[as.character(winner)]
    loser_rating <- ratings[as.character(loser)]
    prob <- expected_elo_score(winner_rating, loser_rating)
    probs[i] <- ifelse(data$higher_rank_won[i], prob, 1 - prob)
  }
  return(probs)
}

# Calculate probabilities for train, validation, and test sets
train_probs_elo <- calculate_elo_probs(train_set, elo_ratings)
validation_probs_elo <- calculate_elo_probs(validation_set, elo_ratings)
test_probs_elo <- calculate_elo_probs(test_set, elo_ratings)

# Calculate metrics for the Elo model
calculate_metrics <- function(probs, actuals) {
  preds <- ifelse(probs > 0.5, 1, 0)
  accuracy <- mean(preds == actuals)
  log_loss <- -mean(actuals * log(probs) + (1 - actuals) * log(1 - probs), na.rm = TRUE)
  calibration <- sum(probs) / sum(actuals)
  return(list(accuracy = accuracy, log_loss = log_loss, calibration = calibration))
}

# Evaluate Elo model on train, validation, and test sets
elo_train_metrics <- calculate_metrics(train_probs_elo, train_set$higher_rank_won)
elo_validation_metrics <- calculate_metrics(validation_probs_elo, validation_set$higher_rank_won)
elo_test_metrics <- calculate_metrics(test_probs_elo, test_set$higher_rank_won)

# Compile Elo model metrics
elo_model_metrics <- data.frame(
  dataset = c("ELO k-factor Training", "ELO k-factor Validation", "ELO k-factor Testing"),
  accuracy = c(elo_train_metrics$accuracy, elo_validation_metrics$accuracy, elo_test_metrics$accuracy),
  log_loss = c(elo_train_metrics$log_loss, elo_validation_metrics$log_loss, elo_test_metrics$log_loss),
  calibration = c(elo_train_metrics$calibration, elo_validation_metrics$calibration, elo_test_metrics$calibration)
)

# Print Elo model performance metrics
print(kable(elo_model_metrics, caption = "ELO Model Performance Metrics for elo k-factor") %>%
        kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                      full_width = F, 
                      position = "center"))


















# Get the list of all players
all_players <- unique(c(train_set$Winner, train_set$Loser, validation_set$Winner, validation_set$Loser, test_set$Winner, test_set$Loser))

# Store performance metrics for each k-value
results <- data.frame(k_value = numeric(), 
                      train_accuracy = numeric(), 
                      train_log_loss = numeric(), 
                      train_calibration = numeric(),
                      test_accuracy = numeric(), 
                      test_log_loss = numeric(), 
                      test_calibration = numeric())

# Define the range of k-values
k_values <- seq(10, 50, by = 5)

# Evaluate performance for each k-value
for (k in k_values) {
  # Initialize Elo ratings
  elo_ratings <- setNames(rep(1500, length(all_players)), all_players)
  
  # Update Elo ratings based on training data
  for (i in 1:nrow(train_set)) {
    winner <- train_set$Winner[i]
    loser <- train_set$Loser[i]
    winner_rating <- elo_ratings[as.character(winner)]
    loser_rating <- elo_ratings[as.character(loser)]
    updated_ratings <- update_elo_ratings(winner_rating, loser_rating, k)
    elo_ratings[as.character(winner)] <- updated_ratings[1]
    elo_ratings[as.character(loser)] <- updated_ratings[2]
  }
  
  # Calculate expected probabilities for training data
  train_probs_elo <- calculate_elo_probs(train_set, elo_ratings)
  # Calculate expected probabilities for test data
  test_probs_elo <- calculate_elo_probs(test_set, elo_ratings)
  
  # Calculate performance metrics
  train_metrics <- calculate_metrics(train_probs_elo, train_set$higher_rank_won)
  test_metrics <- calculate_metrics(test_probs_elo, test_set$higher_rank_won)
  
  # Store results in the data frame
  results <- rbind(results, data.frame(
    k_value = k, 
    train_accuracy = train_metrics$accuracy, 
    train_log_loss = train_metrics$log_loss, 
    train_calibration = train_metrics$calibration,
    test_accuracy = test_metrics$accuracy, 
    test_log_loss = test_metrics$log_loss, 
    test_calibration = test_metrics$calibration
  ))
}

# Print results in a table format
print(results)

# Optionally, save the results to a CSV file for further analysis
write.csv(results, "elo_results.csv", row.names = FALSE)








# Best k-value based on analysis
best_k <- 10

# Initialize Elo ratings
elo_ratings <- setNames(rep(1500, length(all_players)), all_players)

# Update Elo ratings based on training data with the best k-value
for (i in 1:nrow(train_set)) {
  winner <- train_set$Winner[i]
  loser <- train_set$Loser[i]
  winner_rating <- elo_ratings[as.character(winner)]
  loser_rating <- elo_ratings[as.character(loser)]
  updated_ratings <- update_elo_ratings(winner_rating, loser_rating, best_k)
  elo_ratings[as.character(winner)] <- updated_ratings[1]
  elo_ratings[as.character(loser)] <- updated_ratings[2]
}

# Calculate expected probabilities for test data with the best k-value
final_test_probs_elo <- calculate_elo_probs(test_set, elo_ratings)

# Calculate final performance metrics
final_elo_test_metrics <- calculate_metrics(final_test_probs_elo, test_set$higher_rank_won)

# Compile final Elo model metrics
final_elo_model_metrics <- data.frame(
  dataset = "Final ELO Testing",
  accuracy = final_elo_test_metrics$accuracy,
  log_loss = final_elo_test_metrics$log_loss,
  calibration = final_elo_test_metrics$calibration
)

# Append final Elo model metrics to overall performance metrics
performance_metrics <- rbind(performance_metrics, final_elo_model_metrics)

# Print updated performance metrics
print(performance_metrics)






# Load necessary libraries
library(dplyr)

# Define the best k-value based on analysis
best_k <- 10

# Initialize Elo ratings
elo_ratings <- setNames(rep(1500, length(all_players)), all_players)

# Update Elo ratings based on training data with the best k-value
for (i in 1:nrow(train_set)) {
  winner <- train_set$Winner[i]
  loser <- train_set$Loser[i]
  winner_rating <- elo_ratings[as.character(winner)]
  loser_rating <- elo_ratings[as.character(loser)]
  updated_ratings <- update_elo_ratings(winner_rating, loser_rating, best_k)
  elo_ratings[as.character(winner)] <- updated_ratings[1]
  elo_ratings[as.character(loser)] <- updated_ratings[2]
}

# Calculate expected probabilities for test data with the best k-value
final_test_probs_elo <- calculate_elo_probs(test_set, elo_ratings)

# Calculate final performance metrics
final_elo_test_metrics <- calculate_metrics(final_test_probs_elo, test_set$higher_rank_won)

# Compile final Elo model metrics
final_elo_model_metrics <- data.frame(
  dataset = "Final ELO Testing",
  accuracy = final_elo_test_metrics$accuracy,
  log_loss = final_elo_test_metrics$log_loss,
  calibration = final_elo_test_metrics$calibration
)

# Print the final Elo model metrics
print(final_elo_model_metrics)

# Optionally, save the final Elo model metrics to a CSV file for further analysis
write.csv(final_elo_model_metrics, "final_elo_model_metrics.csv", row.names = FALSE)
































# Parameters for advanced Elo model
delta <- 100
nu <- 5
sigma <- 0.1

# Initialize Elo ratings and match counts
elo_ratings_538 <- setNames(rep(1500, length(all_players)), all_players)
match_counts <- setNames(rep(0, length(all_players)), all_players)

# Function to calculate K-factor
calculate_k_factor <- function(matches) {
  delta * (matches + nu) * sigma^2
}

# Function to update Elo ratings with advanced model
update_elo_538 <- function(winner_rating, loser_rating, winner_matches, loser_matches) {
  expected_winner <- expected_elo_score(winner_rating, loser_rating)
  winner_k <- calculate_k_factor(winner_matches)
  loser_k <- calculate_k_factor(loser_matches)
  new_winner_rating <- winner_rating + winner_k * (1 - expected_winner)
  new_loser_rating <- loser_rating + loser_k * (0 - (1 - expected_winner))
  return(c(new_winner_rating, new_loser_rating, winner_matches + 1, loser_matches + 1))
}

# Update advanced Elo ratings based on training data
for (i in 1:nrow(train_set)) {
  winner <- train_set$Winner[i]
  loser <- train_set$Loser[i]
  winner_rating <- elo_ratings_538[as.character(winner)]
  loser_rating <- elo_ratings_538[as.character(loser)]
  winner_matches <- match_counts[as.character(winner)]
  loser_matches <- match_counts[as.character(loser)]
  updated_ratings <- update_elo_538(winner_rating, loser_rating, winner_matches, loser_matches)
  elo_ratings_538[as.character(winner)] <- updated_ratings[1]
  elo_ratings_538[as.character(loser)] <- updated_ratings[2]
  match_counts[as.character(winner)] <- updated_ratings[3]
  match_counts[as.character(loser)] <- updated_ratings[4]
}

# Calculate probabilities for advanced Elo model
calculate_elo_probs_538 <- function(data, ratings) {
  probs <- numeric(nrow(data))
  for (i in 1:nrow(data)) {
    winner <- data$Winner[i]
    loser <- data$Loser[i]
    winner_rating <- ratings[as.character(winner)]
    loser_rating <- ratings[as.character(loser)]
    prob <- expected_elo_score(winner_rating, loser_rating)
    probs[i] <- ifelse(data$higher_rank_won[i], prob, 1 - prob)
  }
  return(probs)
}

# Calculate probabilities for train, validation, and test sets using advanced Elo model
train_probs_elo_538 <- calculate_elo_probs_538(train_set, elo_ratings_538)
validation_probs_elo_538 <- calculate_elo_probs_538(validation_set, elo_ratings_538)
test_probs_elo_538 <- calculate_elo_probs_538(test_set, elo_ratings_538)

# Evaluate advanced Elo model on train, validation, and test sets
elo_538_train_metrics <- calculate_metrics(train_probs_elo_538, train_set$higher_rank_won)
elo_538_validation_metrics <- calculate_metrics(validation_probs_elo_538, validation_set$higher_rank_won)
elo_538_test_metrics <- calculate_metrics(test_probs_elo_538, test_set$higher_rank_won)

# Compile advanced Elo model metrics
elo_538_metrics <- data.frame(
  dataset = c("ELO 538 Training", "ELO 538 Validation", "ELO 538 Testing"),
  accuracy = c(elo_538_train_metrics$accuracy, elo_538_validation_metrics$accuracy, elo_538_test_metrics$accuracy),
  log_loss = c(elo_538_train_metrics$log_loss, elo_538_validation_metrics$log_loss, elo_538_test_metrics$log_loss),
  calibration = c(elo_538_train_metrics$calibration, elo_538_validation_metrics$calibration, elo_538_test_metrics$calibration)
)



# Append advanced Elo model metrics
performance_metrics <- rbind(performance_metrics, elo_538_metrics)

# Print final performance metrics
print(performance_metrics)










library(kableExtra)

# Parameters for advanced Elo model
delta <- 100
nu <- 5
sigma <- 0.1

# Initialize Elo ratings and match counts
elo_ratings_538 <- setNames(rep(1500, length(all_players)), all_players)
match_counts <- setNames(rep(0, length(all_players)), all_players)

# Function to calculate K-factor
calculate_k_factor <- function(matches) {
  delta * (matches + nu) * sigma^2
}

# Function to update Elo ratings with advanced model
update_elo_538 <- function(winner_rating, loser_rating, winner_matches, loser_matches) {
  expected_winner <- expected_elo_score(winner_rating, loser_rating)
  winner_k <- calculate_k_factor(winner_matches)
  loser_k <- calculate_k_factor(loser_matches)
  new_winner_rating <- winner_rating + winner_k * (1 - expected_winner)
  new_loser_rating <- loser_rating + loser_k * (0 - (1 - expected_winner))
  return(c(new_winner_rating, new_loser_rating, winner_matches + 1, loser_matches + 1))
}

# Update advanced Elo ratings based on training data
for (i in 1:nrow(train_set)) {
  winner <- train_set$Winner[i]
  loser <- train_set$Loser[i]
  winner_rating <- elo_ratings_538[as.character(winner)]
  loser_rating <- elo_ratings_538[as.character(loser)]
  winner_matches <- match_counts[as.character(winner)]
  loser_matches <- match_counts[as.character(loser)]
  updated_ratings <- update_elo_538(winner_rating, loser_rating, winner_matches, loser_matches)
  elo_ratings_538[as.character(winner)] <- updated_ratings[1]
  elo_ratings_538[as.character(loser)] <- updated_ratings[2]
  match_counts[as.character(winner)] <- updated_ratings[3]
  match_counts[as.character(loser)] <- updated_ratings[4]
}

# Calculate probabilities for advanced Elo model
calculate_elo_probs_538 <- function(data, ratings) {
  probs <- numeric(nrow(data))
  for (i in 1:nrow(data)) {
    winner <- data$Winner[i]
    loser <- data$Loser[i]
    winner_rating <- ratings[as.character(winner)]
    loser_rating <- ratings[as.character(loser)]
    prob <- expected_elo_score(winner_rating, loser_rating)
    probs[i] <- ifelse(data$higher_rank_won[i], prob, 1 - prob)
  }
  return(probs)
}

# Calculate probabilities for train, validation, and test sets using advanced Elo model
train_probs_elo_538 <- calculate_elo_probs_538(train_set, elo_ratings_538)
validation_probs_elo_538 <- calculate_elo_probs_538(validation_set, elo_ratings_538)
test_probs_elo_538 <- calculate_elo_probs_538(test_set, elo_ratings_538)

# Evaluate advanced Elo model on train, validation, and test sets
elo_538_train_metrics <- calculate_metrics(train_probs_elo_538, train_set$higher_rank_won)
elo_538_validation_metrics <- calculate_metrics(validation_probs_elo_538, validation_set$higher_rank_won)
elo_538_test_metrics <- calculate_metrics(test_probs_elo_538, test_set$higher_rank_won)

# Compile advanced Elo model metrics
elo_538_metrics <- data.frame(
  dataset = c("ELO 538 Training", "ELO 538 Validation", "ELO 538 Testing"),
  accuracy = c(elo_538_train_metrics$accuracy, elo_538_validation_metrics$accuracy, elo_538_test_metrics$accuracy),
  log_loss = c(elo_538_train_metrics$log_loss, elo_538_validation_metrics$log_loss, elo_538_test_metrics$log_loss),
  calibration = c(elo_538_train_metrics$calibration, elo_538_validation_metrics$calibration, elo_538_test_metrics$calibration)
)

# Print 538 ELO model performance metrics
print(kable(elo_538_metrics, caption = "538 ELO Model Performance Metrics for elo 538") %>%
        kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                      full_width = F, 
                      position = "center"))






























# Define the range of values for each parameter
delta_values <- seq(50, 200, by = 50)
nu_values <- seq(1, 10, by = 2)
sigma_values <- seq(0.05, 0.2, by = 0.05)

# Store performance metrics for each combination of parameters
results_538 <- data.frame(delta = numeric(), nu = numeric(), sigma = numeric(), 
                          train_accuracy = numeric(), train_log_loss = numeric(), train_calibration = numeric(),
                          validation_accuracy = numeric(), validation_log_loss = numeric(), validation_calibration = numeric(),
                          test_accuracy = numeric(), test_log_loss = numeric(), test_calibration = numeric())

# Evaluate performance for each combination of parameters
for (delta in delta_values) {
  for (nu in nu_values) {
    for (sigma in sigma_values) {
      
      # Initialize Elo ratings and match counts
      elo_ratings_538 <- setNames(rep(1500, length(all_players)), all_players)
      match_counts <- setNames(rep(0, length(all_players)), all_players)
      
      # Update advanced Elo ratings based on training data
      for (i in 1:nrow(train_set)) {
        winner <- train_set$Winner[i]
        loser <- train_set$Loser[i]
        winner_rating <- elo_ratings_538[as.character(winner)]
        loser_rating <- elo_ratings_538[as.character(loser)]
        winner_matches <- match_counts[as.character(winner)]
        loser_matches <- match_counts[as.character(loser)]
        updated_ratings <- update_elo_538(winner_rating, loser_rating, winner_matches, loser_matches)
        elo_ratings_538[as.character(winner)] <- updated_ratings[1]
        elo_ratings_538[as.character(loser)] <- updated_ratings[2]
        match_counts[as.character(winner)] <- updated_ratings[3]
        match_counts[as.character(loser)] <- updated_ratings[4]
      }
      
      # Calculate probabilities for train, validation, and test sets using advanced Elo model
      train_probs_elo_538 <- calculate_elo_probs_538(train_set, elo_ratings_538)
      validation_probs_elo_538 <- calculate_elo_probs_538(validation_set, elo_ratings_538)
      test_probs_elo_538 <- calculate_elo_probs_538(test_set, elo_ratings_538)
      
      # Evaluate advanced Elo model on train, validation, and test sets
      elo_538_train_metrics <- calculate_metrics(train_probs_elo_538, train_set$higher_rank_won)
      elo_538_validation_metrics <- calculate_metrics(validation_probs_elo_538, validation_set$higher_rank_won)
      elo_538_test_metrics <- calculate_metrics(test_probs_elo_538, test_set$higher_rank_won)
      
      # Store results in the data frame
      results_538 <- rbind(results_538, data.frame(
        delta = delta, nu = nu, sigma = sigma, 
        train_accuracy = elo_538_train_metrics$accuracy, 
        train_log_loss = elo_538_train_metrics$log_loss, 
        train_calibration = elo_538_train_metrics$calibration,
        validation_accuracy = elo_538_validation_metrics$accuracy, 
        validation_log_loss = elo_538_validation_metrics$log_loss, 
        validation_calibration = elo_538_validation_metrics$calibration,
        test_accuracy = elo_538_test_metrics$accuracy, 
        test_log_loss = elo_538_test_metrics$log_loss, 
        test_calibration = elo_538_test_metrics$calibration
      ))
      
      # Print results for each combination of parameters
      print(paste("delta:", delta, "nu:", nu, "sigma:", sigma,
                  "Training Accuracy:", elo_538_train_metrics$accuracy, 
                  "Training Log Loss:", elo_538_train_metrics$log_loss, 
                  "Training Calibration:", elo_538_train_metrics$calibration,
                  "Validation Accuracy:", elo_538_validation_metrics$accuracy, 
                  "Validation Log Loss:", elo_538_validation_metrics$log_loss, 
                  "Validation Calibration:", elo_538_validation_metrics$calibration,
                  "Test Accuracy:", elo_538_test_metrics$accuracy, 
                  "Test Log Loss:", elo_538_test_metrics$log_loss, 
                  "Test Calibration:", elo_538_test_metrics$calibration))
    }
  }
}

# Find the best combination of parameters (based on validation log loss)
best_params <- results_538[which.min(results_538$validation_log_loss), ]

print(paste("The best combination of parameters is: delta =", best_params$delta, 
            ", nu =", best_params$nu, ", sigma =", best_params$sigma))

# Print all results for parameter combinations
print(results_538)






# Define best parameters based on analysis
best_delta <- 50
best_nu <- 1
best_sigma <- 0.05

# Function to calculate K-factor with best parameters
calculate_k_factor_best <- function(matches) {
  best_delta * (matches + best_nu) * best_sigma^2
}

# Function to update Elo ratings with best parameters
update_elo_538_best <- function(winner_rating, loser_rating, winner_matches, loser_matches) {
  expected_winner <- expected_elo_score(winner_rating, loser_rating)
  winner_k <- calculate_k_factor_best(winner_matches)
  loser_k <- calculate_k_factor_best(loser_matches)
  new_winner_rating <- winner_rating + winner_k * (1 - expected_winner)
  new_loser_rating <- loser_rating + loser_k * (0 - (1 - expected_winner))
  return(c(new_winner_rating, new_loser_rating, winner_matches + 1, loser_matches + 1))
}

# Initialize Elo ratings and match counts
elo_ratings_538_best <- setNames(rep(1500, length(all_players)), all_players)
match_counts_best <- setNames(rep(0, length(all_players)), all_players)

# Update advanced Elo ratings based on training data with best parameters
for (i in 1:nrow(train_set)) {
  winner <- train_set$Winner[i]
  loser <- train_set$Loser[i]
  winner_rating <- elo_ratings_538_best[as.character(winner)]
  loser_rating <- elo_ratings_538_best[as.character(loser)]
  winner_matches <- match_counts_best[as.character(winner)]
  loser_matches <- match_counts_best[as.character(loser)]
  updated_ratings <- update_elo_538_best(winner_rating, loser_rating, winner_matches, loser_matches)
  elo_ratings_538_best[as.character(winner)] <- updated_ratings[1]
  elo_ratings_538_best[as.character(loser)] <- updated_ratings[2]
  match_counts_best[as.character(winner)] <- updated_ratings[3]
  match_counts_best[as.character(loser)] <- updated_ratings[4]
}

# Calculate probabilities for train, validation, and test sets using best advanced Elo model
train_probs_elo_538_best <- calculate_elo_probs_538(train_set, elo_ratings_538_best)
validation_probs_elo_538_best <- calculate_elo_probs_538(validation_set, elo_ratings_538_best)
test_probs_elo_538_best <- calculate_elo_probs_538(test_set, elo_ratings_538_best)

# Evaluate advanced Elo model on train, validation, and test sets
elo_538_train_metrics_best <- calculate_metrics(train_probs_elo_538_best, train_set$higher_rank_won)
elo_538_validation_metrics_best <- calculate_metrics(validation_probs_elo_538_best, validation_set$higher_rank_won)
elo_538_test_metrics_best <- calculate_metrics(test_probs_elo_538_best, test_set$higher_rank_won)

# Compile advanced Elo model metrics
elo_538_metrics_best <- data.frame(
  dataset = c("ELO 538 Training (Best)", "ELO 538 Validation (Best)", "ELO 538 Testing (Best)"),
  accuracy = c(elo_538_train_metrics_best$accuracy, elo_538_validation_metrics_best$accuracy, elo_538_test_metrics_best$accuracy),
  log_loss = c(elo_538_train_metrics_best$log_loss, elo_538_validation_metrics_best$log_loss, elo_538_test_metrics_best$log_loss),
  calibration = c(elo_538_train_metrics_best$calibration, elo_538_validation_metrics_best$calibration, elo_538_test_metrics_best$calibration)
)

# Print the advanced Elo model metrics with best parameters in a separate table
print(elo_538_metrics_best)

# Optionally, save the advanced Elo model metrics to a CSV file for further analysis
write.csv(elo_538_metrics_best, "elo_538_model_metrics_best.csv", row.names = FALSE)











# Function to get top N players based on Elo ratings at a given time
get_top_n_players <- function(ratings, n) {
  top_players <- names(sort(ratings, decreasing = TRUE)[1:n])
  return(top_players)
}

# Function to filter matches involving top N players
filter_matches_by_top_players <- function(data, top_players) {
  filtered_data <- data %>%
    filter(Winner %in% top_players | Loser %in% top_players)
  return(filtered_data)
}

# Initialize Elo ratings and match counts
elo_ratings <- setNames(rep(1500, length(all_players)), all_players)
match_counts <- setNames(rep(0, length(all_players)), all_players)

# Update Elo ratings based on the initial training data
for (i in 1:nrow(train_set)) {
  winner <- train_set$Winner[i]
  loser <- train_set$Loser[i]
  winner_rating <- elo_ratings[as.character(winner)]
  loser_rating <- elo_ratings[as.character(loser)]
  winner_matches <- match_counts[as.character(winner)]
  loser_matches <- match_counts[as.character(loser)]
  updated_ratings <- update_elo_538_best(winner_rating, loser_rating, winner_matches, loser_matches)
  elo_ratings[as.character(winner)] <- updated_ratings[1]
  elo_ratings[as.character(loser)] <- updated_ratings[2]
  match_counts[as.character(winner)] <- updated_ratings[3]
  match_counts[as.character(loser)] <- updated_ratings[4]
}

# Get top 50 and top 100 players based on current Elo ratings
top_50_players <- get_top_n_players(elo_ratings, 50)
top_100_players <- get_top_n_players(elo_ratings, 100)

# Filter matches involving top 50 and top 100 players
train_set_top_50 <- filter_matches_by_top_players(train_set, top_50_players)
validation_set_top_50 <- filter_matches_by_top_players(validation_set, top_50_players)
test_set_top_50 <- filter_matches_by_top_players(test_set, top_50_players)

train_set_top_100 <- filter_matches_by_top_players(train_set, top_100_players)
validation_set_top_100 <- filter_matches_by_top_players(validation_set, top_100_players)
test_set_top_100 <- filter_matches_by_top_players(test_set, top_100_players)

# Function to evaluate algorithm on a subset of data
evaluate_algorithm <- function(train_data, validation_data, test_data, ratings, update_func, prob_func) {
  # Update Elo ratings based on training data
  for (i in 1:nrow(train_data)) {
    winner <- train_data$Winner[i]
    loser <- train_data$Loser[i]
    winner_rating <- ratings[as.character(winner)]
    loser_rating <- ratings[as.character(loser)]
    winner_matches <- match_counts[as.character(winner)]
    loser_matches <- match_counts[as.character(loser)]
    updated_ratings <- update_func(winner_rating, loser_rating, winner_matches, loser_matches)
    ratings[as.character(winner)] <- updated_ratings[1]
    ratings[as.character(loser)] <- updated_ratings[2]
    match_counts[as.character(winner)] <- updated_ratings[3]
    match_counts[as.character(loser)] <- updated_ratings[4]
  }
  
  # Calculate probabilities for train, validation, and test sets
  train_probs <- prob_func(train_data, ratings)
  validation_probs <- prob_func(validation_data, ratings)
  test_probs <- prob_func(test_data, ratings)
  
  # Evaluate performance
  train_metrics <- calculate_metrics(train_probs, train_data$higher_rank_won)
  validation_metrics <- calculate_metrics(validation_probs, validation_data$higher_rank_won)
  test_metrics <- calculate_metrics(test_probs, test_data$higher_rank_won)
  
  return(list(train = train_metrics, validation = validation_metrics, test = test_metrics))
}

# Evaluate the advanced Elo model on top 50 and top 100 players
elo_538_metrics_top_50 <- evaluate_algorithm(train_set_top_50, validation_set_top_50, test_set_top_50, elo_ratings, update_elo_538_best, calculate_elo_probs_538)
elo_538_metrics_top_100 <- evaluate_algorithm(train_set_top_100, validation_set_top_100, test_set_top_100, elo_ratings, update_elo_538_best, calculate_elo_probs_538)

# Compile results
results_top_50 <- data.frame(
  dataset = c("ELO 538 Top 50 Training", "ELO 538 Top 50 Validation", "ELO 538 Top 50 Testing"),
  accuracy = c(elo_538_metrics_top_50$train$accuracy, elo_538_metrics_top_50$validation$accuracy, elo_538_metrics_top_50$test$accuracy),
  log_loss = c(elo_538_metrics_top_50$train$log_loss, elo_538_metrics_top_50$validation$log_loss, elo_538_metrics_top_50$test$log_loss),
  calibration = c(elo_538_metrics_top_50$train$calibration, elo_538_metrics_top_50$validation$calibration, elo_538_metrics_top_50$test$calibration)
)

results_top_100 <- data.frame(
  dataset = c("ELO 538 Top 100 Training", "ELO 538 Top 100 Validation", "ELO 538 Top 100 Testing"),
  accuracy = c(elo_538_metrics_top_100$train$accuracy, elo_538_metrics_top_100$validation$accuracy, elo_538_metrics_top_100$test$accuracy),
  log_loss = c(elo_538_metrics_top_100$train$log_loss, elo_538_metrics_top_100$validation$log_loss, elo_538_metrics_top_100$test$log_loss),
  calibration = c(elo_538_metrics_top_100$train$calibration, elo_538_metrics_top_100$validation$calibration, elo_538_metrics_top_100$test$calibration)
)

# Append results to overall performance metrics
performance_metrics <- rbind(performance_metrics, results_top_50, results_top_100)

# Print final performance metrics
print(performance_metrics)















# Function to get top N players based on Elo ratings at a given time
get_top_n_players <- function(ratings, n) {
  top_players <- names(sort(ratings, decreasing = TRUE)[1:n])
  return(top_players)
}

# Function to filter matches involving top N players
filter_matches_by_top_players <- function(data, top_players) {
  filtered_data <- data %>%
    filter(Winner %in% top_players | Loser %in% top_players)
  return(filtered_data)
}

# Initialize Elo ratings and match counts
elo_ratings <- setNames(rep(1500, length(all_players)), all_players)
match_counts <- setNames(rep(0, length(all_players)), all_players)

# Update Elo ratings based on the initial training data
for (i in 1:nrow(train_set)) {
  winner <- train_set$Winner[i]
  loser <- train_set$Loser[i]
  winner_rating <- elo_ratings[as.character(winner)]
  loser_rating <- elo_ratings[as.character(loser)]
  winner_matches <- match_counts[as.character(winner)]
  loser_matches <- match_counts[as.character(loser)]
  updated_ratings <- update_elo_538_best(winner_rating, loser_rating, winner_matches, loser_matches)
  elo_ratings[as.character(winner)] <- updated_ratings[1]
  elo_ratings[as.character(loser)] <- updated_ratings[2]
  match_counts[as.character(winner)] <- updated_ratings[3]
  match_counts[as.character(loser)] <- updated_ratings[4]
}

# Get top 50 and top 100 players based on current Elo ratings
top_50_players <- get_top_n_players(elo_ratings, 50)
top_100_players <- get_top_n_players(elo_ratings, 100)

# Filter matches involving top 50 and top 100 players
train_set_top_50 <- filter_matches_by_top_players(train_set, top_50_players)
validation_set_top_50 <- filter_matches_by_top_players(validation_set, top_50_players)
test_set_top_50 <- filter_matches_by_top_players(test_set, top_50_players)

train_set_top_100 <- filter_matches_by_top_players(train_set, top_100_players)
validation_set_top_100 <- filter_matches_by_top_players(validation_set, top_100_players)
test_set_top_100 <- filter_matches_by_top_players(test_set, top_100_players)

# Function to evaluate algorithm on a subset of data
evaluate_algorithm <- function(train_data, validation_data, test_data, ratings, update_func, prob_func) {
  # Update Elo ratings based on training data
  for (i in 1:nrow(train_data)) {
    winner <- train_data$Winner[i]
    loser <- train_data$Loser[i]
    winner_rating <- ratings[as.character(winner)]
    loser_rating <- ratings[as.character(loser)]
    winner_matches <- match_counts[as.character(winner)]
    loser_matches <- match_counts[as.character(loser)]
    updated_ratings <- update_func(winner_rating, loser_rating, winner_matches, loser_matches)
    ratings[as.character(winner)] <- updated_ratings[1]
    ratings[as.character(loser)] <- updated_ratings[2]
    match_counts[as.character(winner)] <- updated_ratings[3]
    match_counts[as.character(loser)] <- updated_ratings[4]
  }
  
  # Calculate probabilities for train, validation, and test sets
  train_probs <- prob_func(train_data, ratings)
  validation_probs <- prob_func(validation_data, ratings)
  test_probs <- prob_func(test_data, ratings)
  
  # Evaluate performance
  train_metrics <- calculate_metrics(train_probs, train_data$higher_rank_won)
  validation_metrics <- calculate_metrics(validation_probs, validation_data$higher_rank_won)
  test_metrics <- calculate_metrics(test_probs, test_data$higher_rank_won)
  
  return(list(train = train_metrics, validation = validation_metrics, test = test_metrics))
}

# Evaluate the advanced Elo model on top 50 and top 100 players
elo_538_metrics_top_50 <- evaluate_algorithm(train_set_top_50, validation_set_top_50, test_set_top_50, elo_ratings, update_elo_538_best, calculate_elo_probs_538)
elo_538_metrics_top_100 <- evaluate_algorithm(train_set_top_100, validation_set_top_100, test_set_top_100, elo_ratings, update_elo_538_best, calculate_elo_probs_538)

# Compile results for top 50 players
results_top_50 <- data.frame(
  dataset = c("ELO 538 Top 50 Training", "ELO 538 Top 50 Validation", "ELO 538 Top 50 Testing"),
  accuracy = c(elo_538_metrics_top_50$train$accuracy, elo_538_metrics_top_50$validation$accuracy, elo_538_metrics_top_50$test$accuracy),
  log_loss = c(elo_538_metrics_top_50$train$log_loss, elo_538_metrics_top_50$validation$log_loss, elo_538_metrics_top_50$test$log_loss),
  calibration = c(elo_538_metrics_top_50$train$calibration, elo_538_metrics_top_50$validation$calibration, elo_538_metrics_top_50$test$calibration)
)

# Compile results for top 100 players
results_top_100 <- data.frame(
  dataset = c("ELO 538 Top 100 Training", "ELO 538 Top 100 Validation", "ELO 538 Top 100 Testing"),
  accuracy = c(elo_538_metrics_top_100$train$accuracy, elo_538_metrics_top_100$validation$accuracy, elo_538_metrics_top_100$test$accuracy),
  log_loss = c(elo_538_metrics_top_100$train$log_loss, elo_538_metrics_top_100$validation$log_loss, elo_538_metrics_top_100$test$log_loss),
  calibration = c(elo_538_metrics_top_100$train$calibration, elo_538_metrics_top_100$validation$calibration, elo_538_metrics_top_100$test$calibration)
)

# Print the results for top 50 players
print("Performance metrics for ELO 538 Top 50 players:")
print(results_top_50)

# Print the results for top 100 players
print("Performance metrics for ELO 538 Top 100 players:")
print(results_top_100)

# Optionally, save the results to CSV files for further analysis
write.csv(results_top_50, "elo_538_model_metrics_top_50.csv", row.names = FALSE)
write.csv(results_top_100, "elo_538_model_metrics_top_100.csv", row.names = FALSE)


















# # Conclusions:
# # Accuracy:
#   
#   The accuracy for the Top 50 players is higher than for the Top 100 players in both training and validation datasets. Specifically:
#   Training Accuracy: 68.7% (Top 50) vs. 66.2% (Top 100)
# Validation Accuracy: 63.9% (Top 50) vs. 61.7% (Top 100)
# Test Accuracy: 63.6% (Top 50) vs. 63.2% (Top 100)
# This indicates that the model is slightly better at predicting outcomes for the top 50 players compared to the top 100 players.
# Log Loss:
#   
#   The log loss is lower (better) for the Top 50 players in the training set but higher (worse) in the validation set compared to the Top 100 players. Specifically:
#   Training Log Loss: 0.6203 (Top 50) vs. 0.6435 (Top 100)
# Validation Log Loss: 0.6428 (Top 50) vs. 0.6610 (Top 100)
# Test Log Loss: 0.6733 (Top 50) vs. 0.6716 (Top 100)
# Lower log loss indicates better model performance in terms of probability estimation. The model performs slightly worse on the validation set for the top 50 players, suggesting potential overfitting to the training data.
# Calibration:
#   
#   The calibration is generally better (closer to 1) for the Top 100 players than for the Top 50 players. Specifically:
#   Training Calibration: 0.9276 (Top 50) vs. 0.9164 (Top 100)
# Validation Calibration: 1.0349 (Top 50) vs. 0.9934 (Top 100)
# Test Calibration: 0.9565 (Top 50) vs. 0.9144 (Top 100)
# Better calibration indicates that the predicted probabilities are more accurate when compared to actual outcomes. The model is better calibrated for the top 100 players.
# Summary:
#   Predictive Performance: The model is slightly more accurate for the top 50 players but has better overall log loss for the top 100 players.
# Model Calibration: The model's calibration is better for the top 100 players, indicating more reliable probability estimates.
# Overfitting Concern: The lower log loss on the training set and higher on the validation set for the top 50 players suggests potential overfitting when focusing on a smaller subset of players.
# Recommendations:
# Model Tuning: Further tuning and regularization might be required to improve generalization, especially for the top 50 players to avoid overfitting.
# Focus on Calibration: Given the importance of calibration in predictive modeling, efforts should be made to improve calibration, particularly for the top 50 players.
# Additional Evaluation: Perform additional evaluations over different periods to see if these trends hold consistently.
# Overall, these insights can guide improvements in the Elo model and its application across different subsets of players.







# Define function to update Elo ratings with k = 10
update_elo_10 <- function(winner_rating, loser_rating, winner_matches, loser_matches) {
  expected_winner <- expected_elo_score(winner_rating, loser_rating)
  k_factor <- 10
  new_winner_rating <- winner_rating + k_factor * (1 - expected_winner)
  new_loser_rating <- loser_rating + k_factor * (0 - (1 - expected_winner))
  return(c(new_winner_rating, new_loser_rating, winner_matches + 1, loser_matches + 1))
}

# Function to evaluate k = 10 Elo model on a subset of data
evaluate_elo_10 <- function(train_data, validation_data, test_data, ratings, update_func, prob_func) {
  # Initialize Elo ratings and match counts
  ratings <- setNames(rep(1500, length(all_players)), all_players)
  match_counts <- setNames(rep(0, length(all_players)), all_players)
  
  # Update Elo ratings based on training data
  for (i in 1:nrow(train_data)) {
    winner <- train_data$Winner[i]
    loser <- train_data$Loser[i]
    winner_rating <- ratings[as.character(winner)]
    loser_rating <- ratings[as.character(loser)]
    winner_matches <- match_counts[as.character(winner)]
    loser_matches <- match_counts[as.character(loser)]
    updated_ratings <- update_func(winner_rating, loser_rating, winner_matches, loser_matches)
    ratings[as.character(winner)] <- updated_ratings[1]
    ratings[as.character(loser)] <- updated_ratings[2]
    match_counts[as.character(winner)] <- updated_ratings[3]
    match_counts[as.character(loser)] <- updated_ratings[4]
  }
  
  # Calculate probabilities for train, validation, and test sets
  train_probs <- prob_func(train_data, ratings)
  validation_probs <- prob_func(validation_data, ratings)
  test_probs <- prob_func(test_data, ratings)
  
  # Evaluate performance
  train_metrics <- calculate_metrics(train_probs, train_data$higher_rank_won)
  validation_metrics <- calculate_metrics(validation_probs, validation_data$higher_rank_won)
  test_metrics <- calculate_metrics(test_probs, test_data$higher_rank_won)
  
  return(list(train = train_metrics, validation = validation_metrics, test = test_metrics))
}

# Get top 50 and top 100 players based on current Elo ratings
top_50_players <- get_top_n_players(elo_ratings, 50)
top_100_players <- get_top_n_players(elo_ratings, 100)

# Filter matches involving top 50 and top 100 players
train_set_top_50 <- filter_matches_by_top_players(train_set, top_50_players)
validation_set_top_50 <- filter_matches_by_top_players(validation_set, top_50_players)
test_set_top_50 <- filter_matches_by_top_players(test_set, top_50_players)

train_set_top_100 <- filter_matches_by_top_players(train_set, top_100_players)
validation_set_top_100 <- filter_matches_by_top_players(validation_set, top_100_players)
test_set_top_100 <- filter_matches_by_top_players(test_set, top_100_players)

# Evaluate the k = 10 Elo model on top 50 and top 100 players
elo_10_metrics_top_50 <- evaluate_elo_10(train_set_top_50, validation_set_top_50, test_set_top_50, elo_ratings, update_elo_10, calculate_elo_probs_538)
elo_10_metrics_top_100 <- evaluate_elo_10(train_set_top_100, validation_set_top_100, test_set_top_100, elo_ratings, update_elo_10, calculate_elo_probs_538)

# Compile results
results_elo_10_top_50 <- data.frame(
  dataset = c("ELO 10 Top 50 Training", "ELO 10 Top 50 Validation", "ELO 10 Top 50 Testing"),
  accuracy = c(elo_10_metrics_top_50$train$accuracy, elo_10_metrics_top_50$validation$accuracy, elo_10_metrics_top_50$test$accuracy),
  log_loss = c(elo_10_metrics_top_50$train$log_loss, elo_10_metrics_top_50$validation$log_loss, elo_10_metrics_top_50$test$log_loss),
  calibration = c(elo_10_metrics_top_50$train$calibration, elo_10_metrics_top_50$validation$calibration, elo_10_metrics_top_50$test$calibration)
)

results_elo_10_top_100 <- data.frame(
  dataset = c("ELO 10 Top 100 Training", "ELO 10 Top 100 Validation", "ELO 10 Top 100 Testing"),
  accuracy = c(elo_10_metrics_top_100$train$accuracy, elo_10_metrics_top_100$validation$accuracy, elo_10_metrics_top_100$test$accuracy),
  log_loss = c(elo_10_metrics_top_100$train$log_loss, elo_10_metrics_top_100$validation$log_loss, elo_10_metrics_top_100$test$log_loss),
  calibration = c(elo_10_metrics_top_100$train$calibration, elo_10_metrics_top_100$validation$calibration, elo_10_metrics_top_100$test$calibration)
)

# Append results to overall performance metrics
performance_metrics <- rbind(performance_metrics, results_elo_10_top_50, results_elo_10_top_100)

# Print final performance metrics
print(performance_metrics)




# Define function to update Elo ratings with k = 10
update_elo_10 <- function(winner_rating, loser_rating, winner_matches, loser_matches) {
  expected_winner <- expected_elo_score(winner_rating, loser_rating)
  k_factor <- 10
  new_winner_rating <- winner_rating + k_factor * (1 - expected_winner)
  new_loser_rating <- loser_rating + k_factor * (0 - (1 - expected_winner))
  return(c(new_winner_rating, new_loser_rating, winner_matches + 1, loser_matches + 1))
}

# Function to evaluate k = 10 Elo model on a subset of data
evaluate_elo_10 <- function(train_data, validation_data, test_data, update_func, prob_func) {
  # Initialize Elo ratings and match counts
  ratings <- setNames(rep(1500, length(all_players)), all_players)
  match_counts <- setNames(rep(0, length(all_players)), all_players)
  
  # Update Elo ratings based on training data
  for (i in 1:nrow(train_data)) {
    winner <- train_data$Winner[i]
    loser <- train_data$Loser[i]
    winner_rating <- ratings[as.character(winner)]
    loser_rating <- ratings[as.character(loser)]
    winner_matches <- match_counts[as.character(winner)]
    loser_matches <- match_counts[as.character(loser)]
    updated_ratings <- update_func(winner_rating, loser_rating, winner_matches, loser_matches)
    ratings[as.character(winner)] <- updated_ratings[1]
    ratings[as.character(loser)] <- updated_ratings[2]
    match_counts[as.character(winner)] <- updated_ratings[3]
    match_counts[as.character(loser)] <- updated_ratings[4]
  }
  
  # Calculate probabilities for train, validation, and test sets
  train_probs <- prob_func(train_data, ratings)
  validation_probs <- prob_func(validation_data, ratings)
  test_probs <- prob_func(test_data, ratings)
  
  # Evaluate performance
  train_metrics <- calculate_metrics(train_probs, train_data$higher_rank_won)
  validation_metrics <- calculate_metrics(validation_probs, validation_data$higher_rank_won)
  test_metrics <- calculate_metrics(test_probs, test_data$higher_rank_won)
  
  return(list(train = train_metrics, validation = validation_metrics, test = test_metrics))
}

# Get top 50 and top 100 players based on current Elo ratings
top_50_players <- get_top_n_players(elo_ratings, 50)
top_100_players <- get_top_n_players(elo_ratings, 100)

# Filter matches involving top 50 and top 100 players
train_set_top_50 <- filter_matches_by_top_players(train_set, top_50_players)
validation_set_top_50 <- filter_matches_by_top_players(validation_set, top_50_players)
test_set_top_50 <- filter_matches_by_top_players(test_set, top_50_players)

train_set_top_100 <- filter_matches_by_top_players(train_set, top_100_players)
validation_set_top_100 <- filter_matches_by_top_players(validation_set, top_100_players)
test_set_top_100 <- filter_matches_by_top_players(test_set, top_100_players)

# Evaluate the k = 10 Elo model on top 50 and top 100 players
elo_10_metrics_top_50 <- evaluate_elo_10(train_set_top_50, validation_set_top_50, test_set_top_50, update_elo_10, calculate_elo_probs_538)
elo_10_metrics_top_100 <- evaluate_elo_10(train_set_top_100, validation_set_top_100, test_set_top_100, update_elo_10, calculate_elo_probs_538)

# Compile results for top 50 players
results_elo_10_top_50 <- data.frame(
  dataset = c("ELO 10 Top 50 Training", "ELO 10 Top 50 Validation", "ELO 10 Top 50 Testing"),
  accuracy = c(elo_10_metrics_top_50$train$accuracy, elo_10_metrics_top_50$validation$accuracy, elo_10_metrics_top_50$test$accuracy),
  log_loss = c(elo_10_metrics_top_50$train$log_loss, elo_10_metrics_top_50$validation$log_loss, elo_10_metrics_top_50$test$log_loss),
  calibration = c(elo_10_metrics_top_50$train$calibration, elo_10_metrics_top_50$validation$calibration, elo_10_metrics_top_50$test$calibration)
)

# Compile results for top 100 players
results_elo_10_top_100 <- data.frame(
  dataset = c("ELO 10 Top 100 Training", "ELO 10 Top 100 Validation", "ELO 10 Top 100 Testing"),
  accuracy = c(elo_10_metrics_top_100$train$accuracy, elo_10_metrics_top_100$validation$accuracy, elo_10_metrics_top_100$test$accuracy),
  log_loss = c(elo_10_metrics_top_100$train$log_loss, elo_10_metrics_top_100$validation$log_loss, elo_10_metrics_top_100$test$log_loss),
  calibration = c(elo_10_metrics_top_100$train$calibration, elo_10_metrics_top_100$validation$calibration, elo_10_metrics_top_100$test$calibration)
)

# Print the results for top 50 players
print("Performance metrics for ELO 10 Top 50 players:")
print(results_elo_10_top_50)

# Print the results for top 100 players
print("Performance metrics for ELO 10 Top 100 players:")
print(results_elo_10_top_100)

# Optionally, save the results to CSV files for further analysis
write.csv(results_elo_10_top_50, "elo_10_model_metrics_top_50.csv", row.names = FALSE)
write.csv(results_elo_10_top_100, "elo_10_model_metrics_top_100.csv", row.names = FALSE)

















# Conclusions:
#   Accuracy:
#   
#   For the top 50 players, the k = 25 Elo model has higher accuracy than the advanced Elo model (538):
#   Training Accuracy: 70.5% (k = 25) vs. 68.7% (538)
# Validation Accuracy: 66.1% (k = 25) vs. 63.9% (538)
# Test Accuracy: 64.2% (k = 25) vs. 63.6% (538)
# For the top 100 players, the k = 25 Elo model also has higher accuracy:
#   Training Accuracy: 68.3% (k = 25) vs. 66.2% (538)
# Validation Accuracy: 62.9% (k = 25) vs. 61.7% (538)
# Test Accuracy: 62.8% (k = 25) vs. 63.2% (538)
# Log Loss:
#   
#   For the top 50 players, the k = 25 Elo model has lower log loss than the advanced Elo model (538):
#   Training Log Loss: 0.5810 (k = 25) vs. 0.6203 (538)
# Validation Log Loss: 0.6241 (k = 25) vs. 0.6428 (538)
# Test Log Loss: 0.6451 (k = 25) vs. 0.6733 (538)
# For the top 100 players, the k = 25 Elo model has lower log loss:
#   Training Log Loss: 0.6010 (k = 25) vs. 0.6435 (538)
# Validation Log Loss: 0.6439 (k = 25) vs. 0.6610 (538)
# Test Log Loss: 0.6518 (k = 25) vs. 0.6716 (538)
# Calibration:
#   
#   For the top 50 players, the k = 25 Elo model has slightly worse calibration than the advanced Elo model (538):
#   Training Calibration: 0.9408 (k = 25) vs. 0.9276 (538)
# Validation Calibration: 1.0051 (k = 25) vs. 1.0349 (538)
# Test Calibration: 0.9369 (k = 25) vs. 0.9565 (538)
# For the top 100 players, the k = 25 Elo model has better calibration:
#   Training Calibration: 0.9389 (k = 25) vs. 0.9164 (538)
# Validation Calibration: 0.9843 (k = 25) vs. 0.9934 (538)
# Test Calibration: 0.9050 (k = 25) vs. 0.9144 (538)
# Summary:
#   Predictive Performance:
#   
#   The k = 25 Elo model has higher accuracy and lower log loss than the advanced Elo model (538) for both the top 50 and top 100 players. This suggests that the k = 25 Elo model is better at predicting match outcomes for these subsets of players.
# Model Calibration:
#   
#   The k = 25 Elo model has slightly worse calibration for the top 50 players but better calibration for the top 100 players compared to the advanced Elo model (538). This indicates that the k = 25 model's predicted probabilities are more reliable for the top 100 players.
# Recommendations:
# Use k = 25 Elo Model:
# 
# Given the higher accuracy and lower log loss, the k = 25 Elo model appears to be a better choice for predicting match outcomes for both the top 50 and top 100 players.
# Further Calibration Tuning:
# 
# To improve calibration, further tuning or regularization might be necessary, especially for the top 50 players where calibration is slightly worse.
# Periodic Evaluation:
# 
# Periodically evaluate and update the model to ensure it continues to perform well over time as the set of top players changes.
# Overall, the k = 25 Elo model demonstrates superior predictive performance compared to the advanced Elo model (538) across different subsets of players.
# 










