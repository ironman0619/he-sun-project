library(readr)
library(dplyr)
library(ggplot2)
library(knitr)

# Read the dataset containing odds
odds_df <- read_csv("combined_tennis_data_2013_2022.csv")

# Select relevant columns
odds_columns <- c('Date', 'Winner', 'Loser', 'PSW', 'PSL', 'B365W', 'B365L')
odds_df <- odds_df %>% select(all_of(odds_columns))

# Read the original dataset without odds
matches_df <- read_csv("atp_matches_2013.csv")

# Ensure date formats are consistent
matches_df <- matches_df %>%
  mutate(tourney_date = as.Date(as.character(tourney_date), format="%Y%m%d"))

odds_df <- odds_df %>%
  mutate(Date = as.Date(Date, format="%Y-%m-%d"))

# Merge datasets
merged_df <- matches_df %>%
  left_join(odds_df, by=c("tourney_date" = "Date", "winner_name" = "Winner", "loser_name" = "Loser"))

# Ensure the 'higher_rank_won' column exists
if (!"higher_rank_won" %in% colnames(merged_df)) {
  stop("The column 'higher_rank_won' is not present in the merged dataset.")
}

# Split data into training and testing sets
set.seed(123)
train_indices <- sample(seq_len(nrow(merged_df)), size = 0.7 * nrow(merged_df))
matches_train_df <- merged_df[train_indices, ]
matches_test_df <- merged_df[-train_indices, ]

# Sample sizes of training and testing sets
N_train <- nrow(matches_train_df)
N_test <- nrow(matches_test_df)




# Naive model for training set
naive_accuracy_train <- mean(matches_train_df$higher_rank_won)
w_train <- matches_train_df$higher_rank_won

pi_naive_train <- naive_accuracy_train
log_loss_naive_train <- -1 / N_train * sum(w_train * log(pi_naive_train) + (1 - w_train) * log(1 - pi_naive_train))
calibration_naive_train <- pi_naive_train * N_train / sum(w_train)
validation_stats_train <- tibble(model = "Naive Train", pred_acc = naive_accuracy_train, log_loss = log_loss_naive_train, calibration = calibration_naive_train)

# Naive model for testing set
naive_accuracy_test <- mean(matches_test_df$higher_rank_won)
w_test <- matches_test_df$higher_rank_won

pi_naive_test <- naive_accuracy_test
log_loss_naive_test <- -1 / N_test * sum(w_test * log(pi_naive_test) + (1 - w_test) * log(1 - pi_naive_test))
calibration_naive_test <- pi_naive_test * N_test / sum(w_test)
validation_stats_test <- tibble(model = "Naive Test", pred_acc = naive_accuracy_test, log_loss = log_loss_naive_test, calibration = calibration_naive_test)

# Combine validation stats
validation_stats <- bind_rows(validation_stats_train, validation_stats_test)

# Display the validation statistics
kable(validation_stats)




# Logistic model
fit_diff_train <- glm(higher_rank_won ~ diff + 0, data = matches_train_df, family = binomial(link = 'logit'))
summary(fit_diff_train)

tmp_diff_train <- tibble(diff = c(0:10000))
prob_diff_train <- tibble(prob = predict(fit_diff_train, tmp_diff_train, type = 'response'))
tmp_df_train <- tibble(diff = tmp_diff_train$diff, prob = prob_diff_train$prob)
ggplot(aes(x = diff, y = prob), data = tmp_df_train) +
  geom_line() +
  xlab("Difference in points between the higher and lower ranked players") +
  ylab("Probability of the higher ranked player winning") +
  theme_bw()

# Training set predictions
probs_of_winning_train <- predict(fit_diff_train, newdata = matches_train_df, type = "response")
preds_logistic_train <- ifelse(probs_of_winning_train > 0.5, 1, 0)
accuracy_logistic_train <- mean(preds_logistic_train == matches_train_df$higher_rank_won)
log_loss_logistic_train <- -1 / N_train * sum(w_train * log(probs_of_winning_train) + (1 - w_train) * log(1 - probs_of_winning_train), na.rm = TRUE)
calibration_logistic_train <- sum(probs_of_winning_train) / sum(w_train)

# Testing set predictions
probs_of_winning_test <- predict(fit_diff_train, newdata = matches_test_df, type = "response")
preds_logistic_test <- ifelse(probs_of_winning_test > 0.5, 1, 0)
accuracy_logistic_test <- mean(preds_logistic_test == matches_test_df$higher_rank_won)
log_loss_logistic_test <- -1 / N_test * sum(w_test * log(probs_of_winning_test) + (1 - w_test) * log(1 - probs_of_winning_test), na.rm = TRUE)
calibration_logistic_test <- sum(probs_of_winning_test) / sum(w_test)

# Combine logistic validation stats
validation_stats_train <- tibble(model = "Logistic Train", pred_acc = accuracy_logistic_train, log_loss = log_loss_logistic_train, calibration = calibration_logistic_train)
validation_stats_test <- tibble(model = "Logistic Test", pred_acc = accuracy_logistic_test, log_loss = log_loss_logistic_test, calibration = calibration_logistic_test)
validation_stats <- bind_rows(validation_stats_train, validation_stats_test)

# Display the validation statistics
kable(validation_stats)





