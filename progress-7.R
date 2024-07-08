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

# Read in the dataset
tennis_data <- read.csv("combined_tennis_data_2013_2022.csv")
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
train_set <- tennis_data %>% filter(year(Date) < 2021)
validation_set <- tennis_data %>% filter(year(Date) >= 2021 & year(Date) < 2022)
test_set <- tennis_data %>% filter(year(Date) == 2022)

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

# Naive model using 2021 data
train_data_naive <- validation_set %>% filter(year(Date) == 2021)

win_prob_2021 <- mean(train_data_naive$higher_rank_won)

preds_2022 <- rep(win_prob_2021, nrow(test_set))
actual_outcomes_2022 <- test_set$higher_rank_won
naive_accuracy <- mean((preds_2022 > 0.5) == actual_outcomes_2022)

calibration_naive <- sum(preds_2022) / sum(actual_outcomes_2022)

log_loss_naive <- -mean(actual_outcomes_2022 * log(win_prob_2021) + 
                          (1 - actual_outcomes_2022) * log(1 - win_prob_2021), na.rm = TRUE)

naive_metrics <- data.frame(
  dataset = "Naive Testing",
  accuracy = naive_accuracy,
  log_loss = log_loss_naive,
  calibration = calibration_naive
)

# Append naive model metrics
performance_metrics <- rbind(performance_metrics, naive_metrics)

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



