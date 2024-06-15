


# Load necessary libraries using pacman for package management
pacman::p_load(ggplot2, ggplotify, gridExtra, knitr, patchwork, tidyverse, dplyr, lubridate, tidyr)

# Print current working directory
cat("Current working directory:", getwd(), "\n")

# Read the combined tennis dataset
tennis_data <- read.csv("combined_tennis_data_2013_2022.csv")

# Define a function to calculate probabilities from betting odds
calc_probs <- function(odds1, odds2) {
  prob1 <- odds2 / (odds1 + odds2)
  prob2 <- odds1 / (odds1 + odds2)
  return(c(prob1, prob2))
}

# Calculate probabilities for each match and add them to the dataset
tennis_data <- tennis_data %>%
  rowwise() %>%
  mutate(
    probs = list(calc_probs(B365W, B365L)), # Calculate probabilities
    P1_prob = probs[1],                    # Extract Player 1 probability
    P2_prob = probs[2]                     # Extract Player 2 probability
  ) %>%
  ungroup()

# Define a function to calculate logit (log-odds)
calc_logit <- function(p) {
  return(log(p / (1 - p)))
}

# Apply the logit function to Player 1's probability
tennis_data <- tennis_data %>%
  mutate(
    logit_P1_prob = calc_logit(P1_prob) # Calculate logit for Player 1 probability
  )

# Compute the average logit probability for Player 1
average_logit_P1 <- mean(tennis_data$logit_P1_prob, na.rm = TRUE)

# Convert the average logit back to probability
consensus_prob_P1 <- exp(average_logit_P1) / (1 + exp(average_logit_P1))

# Print the consensus probability for Player 1 winning
cat("Consensus probability for Player 1 winning:", consensus_prob_P1, "\n")

# Create a binary indicator for Player 1 winning
tennis_data <- tennis_data %>%
  mutate(Player1_won = Winner == Winner) # Placeholder logic for Player 1 win

# Filter out rows with missing probabilities or outcomes
tennis_data <- tennis_data %>%
  filter(!is.na(P1_prob) & !is.na(Player1_won))

# Define a function to calculate the accuracy of predictions
calc_accuracy <- function(predictions, actuals) {
  return(mean((predictions > 0.5) == actuals))
}

# Define a function to calculate the log loss of predictions
calc_log_loss <- function(probs, actuals) {
  return(-mean(actuals * log(probs) + (1 - actuals) * log(1 - probs)))
}

# Define a function to calculate the calibration of predictions
calc_calibration <- function(probs, actuals) {
  return(sum(probs, na.rm = TRUE) / sum(actuals, na.rm = TRUE))
}

# Calculate the accuracy of predictions
accuracy <- calc_accuracy(tennis_data$P1_prob, tennis_data$Player1_won)

# Calculate the log loss of predictions
log_loss <- calc_log_loss(tennis_data$P1_prob, tennis_data$Player1_won)

# Calculate the calibration of predictions
calibration <- calc_calibration(tennis_data$P1_prob, tennis_data$Player1_won)

# Print the results for accuracy, log loss, and calibration
cat("Accuracy of the predictions:", accuracy, "\n")
cat("Log loss of the predictions:", log_loss, "\n")
cat("Calibration of the predictions:", calibration, "\n")