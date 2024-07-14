library(knitr)
library(patchwork)
library(tidyverse)
library(ggplot2)
library(dplyr)



setwd("/Users/hesun/Desktop")


getwd()


files <- str_glue("tennis_atp-master/atp_matches_{2010:2019}.csv")

raw_matches <- files%>%
  map_dfr(function(x) read_csv(x, show_col_types = FALSE))

matches_df <- raw_matches |>
  select(c(
    "tourney_date",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "match_num",
    "winner_id",
    "loser_id",
    "best_of",
    "winner_rank",
    "winner_rank_points",
    "loser_rank",
    "loser_rank_points")) |>
  mutate_at(
    c("tourney_name", "surface","best_of"),
    as.factor) |>
  mutate_at(c("winner_id", "loser_id"), as.integer) |>
  mutate(tourney_date = ymd(tourney_date))

matches_df <- matches_df |>
  mutate(loser_rank = replace_na(loser_rank, 100000)) |>
  mutate(winner_rank = replace_na(winner_rank, 100000))
matches_df <- matches_df |>
  na.omit() |>
  mutate(higher_rank_won = winner_rank < loser_rank) |>
  mutate(higher_rank_points = winner_rank_points * (higher_rank_won) +
           loser_rank_points * (1 - higher_rank_won)) |>
  mutate(lower_rank_points = winner_rank_points * (1 - higher_rank_won) +
           loser_rank_points * (higher_rank_won))

matches_df <- matches_df |>
  mutate(diff = higher_rank_points - lower_rank_points)



split_time <- dmy("01-01-2019")
matches_train_df <- filter(matches_df, tourney_date < split_time)
matches_test_df <- filter(matches_df, tourney_date >= split_time)









# Naive model
# Naive model training
N_train <- nrow(matches_train_df) # Count the number of rows in the training dataset
naive_accuracy_train <- mean(matches_train_df$higher_rank_won) # Calculate the mean accuracy for the training set
w_train <- matches_train_df$higher_rank_won # Indicator variable for higher ranked player winning

pi_naive_train <- naive_accuracy_train # Probability of higher ranked player winning (naive estimate)
log_loss_naive_train <- -1 / N_train * sum(w_train * log(pi_naive_train) +
                                             (1 - w_train) * log(1 - pi_naive_train)) # Calculate log loss for naive model on training data
calibration_naive_train <- pi_naive_train * N_train / sum(w_train) # Calibration value of the naive model on training data
validation_stats_train <- tibble(model = "Naive Train",
                                 pred_acc = naive_accuracy_train,
                                 log_loss = log_loss_naive_train,
                                 calibration = calibration_naive_train) 


# Naive model testing
N_test <- nrow(matches_test_df) # Count the number of rows in the testing dataset
naive_accuracy_test <- mean(matches_test_df$higher_rank_won) # Calculate the mean accuracy for the testing set
w_test <- matches_test_df$higher_rank_won # Indicator variable for higher ranked player winning

pi_naive_test <- naive_accuracy_test # Probability of higher ranked player winning (naive estimate)
log_loss_naive_test <- -1 / N_test * sum(w_test * log(pi_naive_test) +
                                           (1 - w_test) * log(1 - pi_naive_test)) # Calculate log loss for naive model on testing data
calibration_naive_test <- pi_naive_test * N_test / sum(w_test) # Calibration value of the naive model on testing data
validation_stats_test <- tibble(model = "Naive Test",
                                pred_acc = naive_accuracy_test,
                                log_loss = log_loss_naive_test,
                                calibration = calibration_naive_test) 

validation_stats <- bind_rows(validation_stats_train, validation_stats_test) 


kable(validation_stats)















# logistic model
# plot the graph for D, theta and probability
theta_vals <- seq(0.005, 0.02, by = 0.001)
D <- seq(-500, 500, by = 1)
logistic <- function(x, theta){
  probs <- 1 / (1 + exp(-theta * x))
  return(probs)
}
theta <- c()
probs <- c()
D_vec <- c()
for (t in theta_vals){
  probs <- c(probs, logistic(D, t))
  theta <- c(theta, rep(t, length(D)))
  D_vec <- c(D_vec, D)
}
probs_df <- tibble(p = probs, theta = theta, D = D_vec) |>
  mutate(theta = as.factor(theta))
ggplot(aes(x = D, y = p, color = theta), data = probs_df) +
  geom_line() +
  theme_bw()




# Fit a logistic regression model using the difference in ranking points as the predictor
fit_diff_train <- glm(
  higher_rank_won ~ diff + 0,  
  data = matches_train_df,
  family = binomial(link = 'logit')
)
summary(fit_diff_train)


tmp_diff_train <- tibble(diff = c(0:10000))
prob_diff_train <- tibble(prob = predict(fit_diff_train, tmp_diff_train, type = 'response'))
tmp_df_train <- tibble(diff = tmp_diff_train$diff, prob = prob_diff_train$prob)
ggplot(aes(x = diff, y = prob), data = tmp_df_train) +
  geom_line() +
  xlab("difference in points between the higher and lower ranked players") +
  ylab("probability of the higher ranked player winning") +
  theme_bw()

# Calculate predictions, accuracy, log loss, and calibration for the training set
probs_of_winning_train <- predict(fit_diff_train, newdata = matches_train_df, type = "response")  
preds_logistic_train <- ifelse(probs_of_winning_train > 0.5, 1, 0)  
accuracy_logistic_train <- mean(preds_logistic_train == matches_train_df$higher_rank_won)  
w_train <- matches_train_df$higher_rank_won
log_loss_logistic_train <- -1 / N_train * sum(w_train * log(probs_of_winning_train) +
                                                (1 - w_train) * log(1 - probs_of_winning_train), na.rm = TRUE)  
calibration_logistic_train <- sum(probs_of_winning_train) / sum(w_train)  

# Calculate predictions, accuracy, log loss, and calibration for the testing set
probs_of_winning_test <- predict(fit_diff_train, newdata = matches_test_df, type = "response")  
preds_logistic_test <- ifelse(probs_of_winning_test > 0.5, 1, 0)  
accuracy_logistic_test <- mean(preds_logistic_test == matches_test_df$higher_rank_won)  
w_test <- matches_test_df$higher_rank_won
log_loss_logistic_test <- -1 / N_test * sum(w_test * log(probs_of_winning_test) +
                                              (1 - w_test) * log(1 - probs_of_winning_test), na.rm = TRUE)  
calibration_logistic_test <- sum(probs_of_winning_test) / sum(w_test) 


validation_stats_train <- tibble(model = "Logistic Train",
                                 pred_acc = accuracy_logistic_train,
                                 log_loss = log_loss_logistic_train,
                                 calibration = calibration_logistic_train)

validation_stats_test <- tibble(model = "Logistic Test",
                                pred_acc = accuracy_logistic_test,
                                log_loss = log_loss_logistic_test,
                                calibration = calibration_logistic_test)

# Combine validation stats from both training and testing
validation_stats <- bind_rows(validation_stats_train, validation_stats_test)


kable(validation_stats)


























# 25 k-factor elo model
# plot the graph of elo rating for 3 players
# Define a function to calculate the expected score
expected_score <- function(rating_a, rating_b) {
  1 / (1 + 10^((rating_b - rating_a) / 400))  
}

# Define a function to update Elo ratings
update_elo <- function(winner_rating, loser_rating, k = 25) {
  expected_winner <- expected_score(winner_rating, loser_rating)
  expected_loser <- 1 - expected_winner
  
  new_winner_rating <- winner_rating + k * (1 - expected_winner)  # Update winner's rating
  new_loser_rating <- loser_rating - k * (expected_loser)  # Update loser's rating
  
  return(c(new_winner_rating, new_loser_rating))
}

# Initialize Elo ratings
players <- unique(c(matches_train_df$winner_id, matches_train_df$loser_id))  # Extract unique player IDs
elo_ratings <- setNames(rep(1500, length(players)), as.character(players))  # Assign initial Elo rating of 1500 to each player

# Player IDs for specific players
player_ids <- c(104918, 104792, 105227)  # Andy Murray, Gael Monfils, Marin Cilic

# Prepare a dataframe to store Elo ratings and dates
elo_history <- data.frame(Player_ID = integer(), Date = as.Date(character()), Elo = numeric())

# Update Elo ratings and track the ratings of three players
for (i in 1:nrow(matches_train_df)) {
  winner <- matches_train_df$winner_id[i]
  loser <- matches_train_df$loser_id[i]
  
  winner_rating <- elo_ratings[as.character(winner)]
  loser_rating <- elo_ratings[as.character(loser)]
  
  new_ratings <- update_elo(winner_rating, loser_rating)
  elo_ratings[as.character(winner)] <- new_ratings[1]
  elo_ratings[as.character(loser)] <- new_ratings[2]
  
  if (winner %in% player_ids || loser %in% player_ids) {
    player_id <- ifelse(winner %in% player_ids, winner, loser)
    current_elo <- ifelse(winner %in% player_ids, new_ratings[1], new_ratings[2])
    date <- as.Date(matches_train_df$tourney_date[i], format = "%Y%m%d")
    elo_history <- rbind(elo_history, data.frame(Player_ID = player_id, Date = date, Elo = current_elo))
  }
}

# Convert Player_ID to factor for easier plotting
elo_history$Player_ID <- factor(elo_history$Player_ID, levels = player_ids, labels = c("Andy Murray", "Gael Monfils", "Marin Cilic"))

# Plot Elo rating changes for each player
for (player in levels(elo_history$Player_ID)) {
  player_data <- filter(elo_history, Player_ID == player)
  p <- ggplot(player_data, aes(x = Date, y = Elo)) +
    geom_line(color = "blue") +
    labs(title = paste(player, "'s Elo Rating Over Time"), x = "Date", y = "Elo Rating") +
    theme_minimal()
  print(p)  # Display the plot
}

# Assuming the Elo ratings have been accurately updated and stored in elo_history
elo_history$Player_ID <- as.factor(elo_history$Player_ID)

# Create plots for each player's Elo rating changes
unique_players <- unique(elo_history$Player_ID)
plot_list <- list()
for (player in unique_players) {
  plot_list[[as.character(player)]] <- ggplot(subset(elo_history, Player_ID == player), aes(x = Date, y = Elo)) +
    geom_line() +
    labs(title = paste(player, "'s Elo Rating Over Time"), x = "Date", y = "Elo Rating") +
    theme_minimal()
}

# Display the plot
print(plot_list[["Andy Murray"]])
print(plot_list[["Gael Monfils"]])
print(plot_list[["Marin Cilic"]])









# update elo ratings and calculate the validation metric.
# Initialize Elo Ratings with players from both training and test sets
players <- unique(c(matches_train_df$winner_id, matches_train_df$loser_id, 
                    matches_test_df$winner_id, matches_test_df$loser_id))
elo_ratings <- setNames(rep(1500, length(players)), players)

# Function to calculate expected score
expected_score <- function(rating_a, rating_b) {
  1 / (1 + 10^((rating_b - rating_a) / 400))
}

# Function to update Elo ratings
update_elo <- function(winner_rating, loser_rating, k = 25) {
  expected_winner <- expected_score(winner_rating, loser_rating)
  expected_loser <- 1 - expected_winner
  
  new_winner_rating <- winner_rating + k * (1 - expected_winner)
  new_loser_rating <- loser_rating + k * (0 - expected_loser)
  
  return(c(new_winner_rating, new_loser_rating))
}

# Update Elo ratings using the training set
for(i in 1:nrow(matches_train_df)) {
  winner <- matches_train_df$winner_id[i]
  loser <- matches_train_df$loser_id[i]
  
  winner_rating <- elo_ratings[as.character(winner)]
  loser_rating <- elo_ratings[as.character(loser)]
  
  new_ratings <- update_elo(winner_rating, loser_rating)
  
  elo_ratings[as.character(winner)] <- new_ratings[1]
  elo_ratings[as.character(loser)] <- new_ratings[2]
}

# Calculate the training predictions and update Elo ratings after each game
elo_expected_probs_train <- numeric(nrow(matches_train_df))
for(i in 1:nrow(matches_train_df)) {
  winner <- matches_train_df$winner_id[i]
  loser <- matches_train_df$loser_id[i]
  
  # Retrieve current Elo ratings
  winner_rating <- elo_ratings[as.character(winner)]
  loser_rating <- elo_ratings[as.character(loser)]
  
  # Calculate expected probability of winning based on current Elo ratings
  expected_prob <- expected_score(winner_rating, loser_rating)
  
  # Store expected probabilities based on the higher-ranked winner
  if (matches_train_df$higher_rank_won[i]) {
    elo_expected_probs_train[i] <- expected_prob
  } else {
    elo_expected_probs_train[i] <- 1 - expected_prob
  }
  
  # Update Elo ratings based on the match outcome
  new_ratings <- update_elo(winner_rating, loser_rating)
  elo_ratings[as.character(winner)] <- new_ratings[1]
  elo_ratings[as.character(loser)] <- new_ratings[2]
}

actual_outcomes_elo_train <- matches_train_df$higher_rank_won
elo_predictions_train <- ifelse(elo_expected_probs_train > 0.5, 1, 0)
accuracy_elo_train <- mean(elo_predictions_train == actual_outcomes_elo_train)
log_loss_elo_train <- -1 / nrow(matches_train_df) * sum(actual_outcomes_elo_train * log(elo_expected_probs_train) +
                                                          (1 - actual_outcomes_elo_train) * log(1 - elo_expected_probs_train))
calibration_elo_train <- sum(elo_expected_probs_train) / sum(actual_outcomes_elo_train)

# Calculate the testing predictions and update Elo ratings after each game
elo_expected_probs_test <- numeric(nrow(matches_test_df))
for(i in 1:nrow(matches_test_df)) {
  winner <- matches_test_df$winner_id[i]
  loser <- matches_test_df$loser_id[i]
  
  # Retrieve current Elo ratings; if a player is not found, assign the base Elo rating of 1500
  winner_rating <- ifelse(is.na(elo_ratings[as.character(winner)]), 1500, elo_ratings[as.character(winner)])
  loser_rating <- ifelse(is.na(elo_ratings[as.character(loser)]), 1500, elo_ratings[as.character(loser)])
  
  # Calculate expected probability of winning based on current Elo ratings
  if (matches_test_df$higher_rank_won[i]){
    elo_expected_probs_test[i] <- expected_score(winner_rating, loser_rating)
  } else {
    elo_expected_probs_test[i] <- expected_score(loser_rating, winner_rating)
  }
  
  # Update Elo ratings based on the match outcome
  new_ratings <- update_elo(winner_rating, loser_rating)
  elo_ratings[as.character(winner)] <- new_ratings[1]
  elo_ratings[as.character(loser)] <- new_ratings[2]
}

actual_outcomes_elo_test <- matches_test_df$higher_rank_won
elo_predictions_test <- ifelse(elo_expected_probs_test > 0.5, 1, 0)
accuracy_elo_test <- mean(elo_predictions_test == actual_outcomes_elo_test)
log_loss_elo_test <- -1 / nrow(matches_test_df) * sum(actual_outcomes_elo_test * log(elo_expected_probs_test) +
                                                        (1 - actual_outcomes_elo_test) * log(1 - elo_expected_probs_test))
calibration_elo_test <- sum(elo_expected_probs_test) / sum(actual_outcomes_elo_test)

validation_stats <- tibble(
  dataset = c("Train", "Test"),
  model = c("25-k-Elo", "25-k-Elo"),
  pred_acc = c(accuracy_elo_train, accuracy_elo_test),
  log_loss = c(log_loss_elo_train, log_loss_elo_test),
  calibration = c(calibration_elo_train, calibration_elo_test)
)

kable(validation_stats)

















# 538 elo model:
# update elo rating and calculate training and testing set validation mertic

# initialise the elo rating
delta <- 100
nu <- 5
sigma <- 0.1
players <- unique(c(matches_train_df$winner_id, matches_train_df$loser_id, 
                    matches_test_df$winner_id, matches_test_df$loser_id))
elo_ratings_538 <- setNames(rep(1500, length(players)), players)
match_counts <- setNames(rep(0, length(players)), players)

# calculate the expected score
expected_score <- function(rating_a, rating_b) {
  1 / (1 + 10^((rating_b - rating_a) / 400))
}

# calculate K-factor
k_factor_538 <- function(m) {
  delta / (m + nu)^sigma
}


# update elo rating, including k-factor
update_elo_538 <- function(winner_rating, loser_rating, winner_matches, loser_matches) {
  expected_winner <- expected_score(winner_rating, loser_rating)
  expected_loser <- 1 - expected_winner
  
  winner_k <- k_factor_538(winner_matches)
  loser_k <- k_factor_538(loser_matches)
  
  new_winner_rating <- winner_rating + winner_k * (1 - expected_winner)
  new_loser_rating <- loser_rating - loser_k * (expected_loser)
  
  return(c(new_winner_rating, new_loser_rating))
}

# use training set to update elo rating
elo_expected_probs_train <- numeric(nrow(matches_train_df))
for(i in 1:nrow(matches_train_df)) {
  winner <- as.character(matches_train_df$winner_id[i])
  loser <- as.character(matches_train_df$loser_id[i])
  
  winner_matches <- match_counts[winner]
  loser_matches <- match_counts[loser]
  
  # update Elo rating
  new_ratings <- update_elo_538(elo_ratings_538[winner], elo_ratings_538[loser], winner_matches, loser_matches)
  elo_ratings_538[winner] <- new_ratings[1]
  elo_ratings_538[loser] <- new_ratings[2]
  
  match_counts[winner] <- winner_matches + 1
  match_counts[loser] <- loser_matches + 1
  

  expected_prob <- expected_score(new_ratings[1], new_ratings[2])
  elo_expected_probs_train[i] <- ifelse(matches_train_df$higher_rank_won[i], expected_prob, 1 - expected_prob)
}

# calculate 
accuracy_train <- mean(ifelse(elo_expected_probs_train > 0.5, 1, 0) == matches_train_df$higher_rank_won)
log_loss_train <- -mean(log(ifelse(matches_train_df$higher_rank_won, elo_expected_probs_train, 1 - elo_expected_probs_train)))
calibration_train <- sum(elo_expected_probs_train) / sum(matches_train_df$higher_rank_won)

# use testing set to predict, calculate validation metric
elo_expected_probs_test <- numeric(nrow(matches_test_df))
for(i in 1:nrow(matches_test_df)) {
  winner <- as.character(matches_test_df$winner_id[i])
  loser <- as.character(matches_test_df$loser_id[i])
  
  winner_matches <- match_counts[winner]
  loser_matches <- match_counts[loser]
  
  # update elo rating
  new_ratings <- update_elo_538(elo_ratings_538[winner], elo_ratings_538[loser], winner_matches, loser_matches)
  elo_ratings_538[winner] <- new_ratings[1]
  elo_ratings_538[loser] <- new_ratings[2]
  
  match_counts[winner] <- winner_matches + 1
  match_counts[loser] <- loser_matches + 1
  
  # 
  expected_prob <- expected_score(new_ratings[1], new_ratings[2])
  elo_expected_probs_test[i] <- ifelse(matches_test_df$higher_rank_won[i], expected_prob, 1 - expected_prob)
}

# calculate accuarcy, log-loss and calibration
accuracy_test <- mean(ifelse(elo_expected_probs_test > 0.5, 1, 0) == matches_test_df$higher_rank_won)
log_loss_test <- -mean(log(ifelse(matches_test_df$higher_rank_won, elo_expected_probs_test, 1 - elo_expected_probs_test)))
calibration_test <- sum(elo_expected_probs_test) / sum(matches_test_df$higher_rank_won)


validation_stats <- tibble(
  dataset = c("Train", "Test"),
  model = "Elo-538",
  pred_acc = c(accuracy_train, accuracy_test),
  log_loss = c(log_loss_train, log_loss_test),
  calibration = c(calibration_train, calibration_test)
)

kable(validation_stats)










library(tidyverse)
library(lubridate)

# Function to run the Elo-538 model and return performance metrics
run_elo_538 <- function(matches_train_df, matches_test_df, delta, nu, sigma) {
  players <- unique(c(matches_train_df$winner_id, matches_train_df$loser_id, 
                      matches_test_df$winner_id, matches_test_df$loser_id))
  elo_ratings_538 <- setNames(rep(1500, length(players)), players)
  match_counts <- setNames(rep(0, length(players)), players)
  
  expected_score <- function(rating_a, rating_b) {
    1 / (1 + 10^((rating_b - rating_a) / 400))
  }
  
  k_factor_538 <- function(m) {
    delta / (m + nu)^sigma
  }
  
  update_elo_538 <- function(winner_rating, loser_rating, winner_matches, loser_matches) {
    expected_winner <- expected_score(winner_rating, loser_rating)
    expected_loser <- 1 - expected_winner
    
    winner_k <- k_factor_538(winner_matches)
    loser_k <- k_factor_538(loser_matches)
    
    new_winner_rating <- winner_rating + winner_k * (1 - expected_winner)
    new_loser_rating <- loser_rating - loser_k * (expected_loser)
    
    return(c(new_winner_rating, new_loser_rating))
  }
  
  # Training set
  elo_expected_probs_train <- numeric(nrow(matches_train_df))
  for(i in 1:nrow(matches_train_df)) {
    winner <- as.character(matches_train_df$winner_id[i])
    loser <- as.character(matches_train_df$loser_id[i])
    
    winner_matches <- match_counts[winner]
    loser_matches <- match_counts[loser]
    
    new_ratings <- update_elo_538(elo_ratings_538[winner], elo_ratings_538[loser], winner_matches, loser_matches)
    elo_ratings_538[winner] <- new_ratings[1]
    elo_ratings_538[loser] <- new_ratings[2]
    
    match_counts[winner] <- winner_matches + 1
    match_counts[loser] <- loser_matches + 1
    
    expected_prob <- expected_score(new_ratings[1], new_ratings[2])
    elo_expected_probs_train[i] <- ifelse(matches_train_df$higher_rank_won[i], expected_prob, 1 - expected_prob)
  }
  
  accuracy_train <- mean(ifelse(elo_expected_probs_train > 0.5, 1, 0) == matches_train_df$higher_rank_won)
  log_loss_train <- -mean(log(ifelse(matches_train_df$higher_rank_won, elo_expected_probs_train, 1 - elo_expected_probs_train)))
  calibration_train <- sum(elo_expected_probs_train) / sum(matches_train_df$higher_rank_won)
  
  # Testing set
  elo_expected_probs_test <- numeric(nrow(matches_test_df))
  for(i in 1:nrow(matches_test_df)) {
    winner <- as.character(matches_test_df$winner_id[i])
    loser <- as.character(matches_test_df$loser_id[i])
    
    winner_matches <- match_counts[winner]
    loser_matches <- match_counts[loser]
    
    new_ratings <- update_elo_538(elo_ratings_538[winner], elo_ratings_538[loser], winner_matches, loser_matches)
    elo_ratings_538[winner] <- new_ratings[1]
    elo_ratings_538[loser] <- new_ratings[2]
    
    match_counts[winner] <- winner_matches + 1
    match_counts[loser] <- loser_matches + 1
    
    expected_prob <- expected_score(new_ratings[1], new_ratings[2])
    elo_expected_probs_test[i] <- ifelse(matches_test_df$higher_rank_won[i], expected_prob, 1 - expected_prob)
  }
  
  accuracy_test <- mean(ifelse(elo_expected_probs_test > 0.5, 1, 0) == matches_test_df$higher_rank_won)
  log_loss_test <- -mean(log(ifelse(matches_test_df$higher_rank_won, elo_expected_probs_test, 1 - elo_expected_probs_test)))
  calibration_test <- sum(elo_expected_probs_test) / sum(matches_test_df$higher_rank_won)
  
  validation_stats <- tibble(
    delta = delta,
    nu = nu,
    sigma = sigma,
    accuracy_train = accuracy_train,
    log_loss_train = log_loss_train,
    calibration_train = calibration_train,
    accuracy_test = accuracy_test,
    log_loss_test = log_loss_test,
    calibration_test = calibration_test
  )
  
  return(validation_stats)
}

# Grid search over delta, nu, and sigma values
delta_values <- c(50, 100, 150)
nu_values <- c(5, 10, 15)
sigma_values <- c(0.05, 0.1, 0.15)

results <- expand.grid(delta = delta_values, nu = nu_values, sigma = sigma_values) %>%
  pmap_dfr(~run_elo_538(matches_train_df, matches_test_df, ..1, ..2, ..3))

# Display the results
results %>%
  arrange(log_loss_test)


# Display the top results based on log_loss_test
top_results <- results %>%
  arrange(log_loss_test) %>%
  head(10) # Display the top 10 results

print(top_results)


# Display the top results based on accuracy_test
top_accuracy_results <- results %>%
  arrange(desc(accuracy_test)) %>%
  head(10) # Display the top 10 results

print(top_accuracy_results)



# optimised value
optimal_params <- list(delta = 150, nu = 5, sigma = 0.05)

print(optimal_params)













# plot 538 elo rating graph for 3 players
players_of_interest <- c("104918", "104792", "105227")  #  Andy Murray, Gael Monfils, Marin Cilic


elo_ratings_538 <- setNames(rep(1500, length(players)), players)
match_counts <- setNames(rep(0, length(players)), players)
elo_history <- data.frame(Date = integer(), Elo = numeric(), PlayerID = character())


for(i in 1:nrow(matches_train_df)) {
  winner <- as.character(matches_train_df$winner_id[i])
  loser <- as.character(matches_train_df$loser_id[i])
  date <- as.Date(matches_train_df$tourney_date[i], format = "%Y%m%d")
  
  winner_matches <- match_counts[winner]
  loser_matches <- match_counts[loser]
  
  new_ratings <- update_elo_538(elo_ratings_538[winner], elo_ratings_538[loser], winner_matches, loser_matches)
  
  elo_ratings_538[winner] <- new_ratings[1]
  elo_ratings_538[loser] <- new_ratings[2]
  
  match_counts[winner] <- winner_matches + 1
  match_counts[loser] <- loser_matches + 1
  
  
  # record elo rating of the players we need
  if (winner %in% players_of_interest) {
    elo_history <- rbind(elo_history, data.frame(Date = date, Elo = new_ratings[1], PlayerID = winner))
  }
  if (loser %in% players_of_interest) {
    elo_history <- rbind(elo_history, data.frame(Date = date, Elo = new_ratings[2], PlayerID = loser))
  }
}

# plot elo rating graph for every player
plot_list <- list()
for (player_id in players_of_interest) {
  player_data <- elo_history %>% filter(PlayerID == player_id)
  player_name <- case_when(
    player_id == "104918" ~ "Andy Murray",
    player_id == "104792" ~ "Gael Monfils",
    player_id == "105227" ~ "Marin Cilic"
  )
  plot <- ggplot(player_data, aes(x = Date, y = Elo)) +
    geom_line(color = "blue") +
    labs(title = paste(player_name, "Elo Rating Over Time (538 Model)"),
         x = "Date",
         y = "Elo Rating") +
    theme_minimal()
  
  plot_list[[player_name]] <- plot
}


plot_list[["Andy Murray"]]
plot_list[["Gael Monfils"]]
plot_list[["Marin Cilic"]]




























