#title: "MovieLens Recommendation System"
#author: "Fidan Alasgarova"
#date: "07.03.2026"


##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 600)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Data exploratory analsysis

# Number of ratings in the edx and final_holdout_test datasets
dim(edx)
dim(final_holdout_test)

# Number of unique users and unique movies, and max possible number of ratings in the edx dataset  
length(unique(edx$userId))
length(unique(edx$movieId))
length(unique(edx$userId)) * length(unique(edx$movieId))

# Sparsity matrix"
# Build rating matrix
library(Matrix)
library(tidyr)
library(dplyr)

rating_matrix <- edx %>%
  select(userId, movieId, rating, title) %>%
  pivot_wider(
    id_cols = c(movieId, title),
    names_from = userId,
    values_from = rating,
    values_fill = 0)

matrix_subset <- rating_matrix[1:100, 1:100]
matrix_subset <- matrix_subset[, -c(1, 2)]
matrix_subset <- matrix_subset %>%
  as.data.frame() %>%
  mutate(movie = row_number()) %>%
  pivot_longer(
    cols = -movie, 
    names_to = "userId", 
    values_to = "rating")

# Visualize sparsity matrix
matrix_subset %>%
  mutate(rated = rating > 0) %>%
  ggplot(aes(x = userId, y = movie, fill = rated)) +
  geom_tile(color="grey80") +
  scale_fill_manual(values = c("white", "#228B22")) +
  labs(title = "Sparsity Pattern: Rated vs not Rated") +
  theme_minimal() +
  theme(
    axis.text.x  = element_blank())

# Number of duplicate rows in the edx dataset
sum(duplicated(edx))

# Number of duplicate ratings in the edx dataset
edx %>%
  count(userId, movieId) %>%
  filter(n > 1)

# Unique rating values in the edx dataset
sort(unique(edx$rating))

# Global mean in the edx dataset
mu <- mean(edx$rating)
mu

# Distribution of rating values (05. to 5) in the edx dataset
rating_distribution <- as.data.frame(table(edx$rating)) 
colnames(rating_distribution) <- c("Rating", "Count")

ggplot(rating_distribution, aes(x = Rating, y = Count))+
  geom_bar(stat = "identity", fill = "darkslategrey") +
  geom_text(
    aes(label = Count), vjust = -0.3)+
  scale_y_continuous(
    labels = function(x) sprintf("%.1f", x / 1e6))+
  labs(y = "Count (millions)") +
  labs(title = "Distribution of Rating Values")

# Glimpse into edx dataset 
head(edx)
 
# Separated title and year, and reduce genre columns
# Rename edx and final_houldout set
edx_set <- edx
validation_set <- final_holdout_test

# Separate title and year columns in the edx set
edx_set <- edx_set %>%
  mutate(year = as.numeric(str_extract(title, "\\(\\d{4}\\)") %>% 
                             str_remove_all("[()]")),
         title = str_remove(title, "\\s*\\(\\d{4}\\)"))

# Split genre column to keep only the first value in the edx set
edx_set <- edx_set %>%
  mutate(
    genre = str_extract(genres, "^[^|]+")) %>%
  select(-genres)

# Convert timestamp into date format in the edx set 
library(lubridate)
edx_set <- edx_set %>% 
  mutate(review = year(as_datetime(timestamp)))

# Split title and year column in the validation set 
validation_set <- validation_set%>%
  mutate(year = as.numeric(str_extract(title, "\\(\\d{4}\\)") %>% 
                             str_remove_all("[()]")), title = str_remove(title, "\\s*\\(\\d{4}\\)"))

# Split genre column to keep only the first value in the validation set 
validation_set <- validation_set %>%
  mutate(genre = str_extract(genres, "^[^|]+")) %>%
  select(-genres)

# Convert timestamp into date format in the validation set 
validation_set <- validation_set %>% 
  mutate(review = year(as_datetime(timestamp)))

# Glimpse into edx_set dataset 
head(edx_set)

# Create test set 80% and train set 20% from edx dataset
set.seed(1)
train_index <- createDataPartition(y = edx_set$rating, 
                                   times = 1, p = 0.80, list = FALSE)

train <- edx_set[train_index, ]
test  <- edx_set[-train_index, ]

test <- test %>%
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# RMSE based on global mean
RMSE_mu <- RMSE(edx$rating, mu)
RMSE_mu

# Number of ratings per movie in the edx dataset
edx_set %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=30, fill = "darkslategrey", color = I("white")) +
  scale_x_log10() +
  labs(x = "Movies", y = "Number of ratings", title = "Number of Ratings per Movie")

# Adjusting the model for movie effect b_i
b_i <- train %>% 
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# RMSE based on mu, b_i
predicted_b_i <- test %>% 
  left_join(b_i, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

RMSE(test$rating, predicted_b_i)

# Number of ratings per user in the edx dataset
edx_set %>% count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, fill = "darkslategrey", color = I("white")) +
  scale_x_log10() +
  labs(x = "Users", y = "Number of ratings", title = "Number of Ratings per User")

# Adjusting the model for user effect b_u
b_u <- train %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# RMSE based on mu, b_i, b_u
predicted_b_u <- test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

RMSE(test$rating, predicted_b_u)

# Number of ratings per genre in the edx dataset
edx_set %>%
  group_by(genre) %>%
  summarize(ratings = n()) %>%
  arrange(desc(ratings))

# Average rating per genre in the edx dataset
edx_set %>%
  group_by(genre) %>%
  summarize(
    ratings = n(),
    average = mean(rating)) %>%
  arrange(desc(average)) %>%
  mutate(average)

# Adjusting the model for genre effect b_g
b_g <- train %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genre) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

# RMSE based on mu, b_i, b_u and b_g
predicted_b_g <- test %>%
  left_join(b_i, by = 'movieId') %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genre") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

RMSE(test$rating, predicted_b_g)

# Average rating per year of release in the edx dataset
edx_set %>%  
  group_by(year) %>%
  summarize(n = n()) %>%
  arrange(year) %>%
  ggplot(aes(x = year, y = n)) +
  geom_col(fill = "darkslategrey", color = "white") +
  scale_y_continuous(labels = scales::label_number(scale = 1e-3, suffix = "K")) +
  labs(x = "Year", y = "Number of ratings (Thousands)", title = "Number of Ratings by Year of Release")

# Adjusting the model for year of release effect b_y
b_y <- train %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genre") %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u - b_g))

# RMSE based on global mu, b_i, b_u, b_g and b_y effect

predicted_b_y <- test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genre") %>%
  left_join(b_y, by = "year") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y) %>%
  pull(pred)

RMSE(test$rating, predicted_b_y)

# Distribution of ratings based on movie review year in the edx dataset
edx_set %>% group_by(review) %>%
  summarize(n = n()) %>%
  arrange(review) %>%
  ggplot(aes(x=review, y= n)) +
  geom_col(fill = "darkslategrey") +
  labs(x = "Year", y = "Number of ratings", title = "Number of Ratings by Review Year")

# Adjusting the model for review date effect (b_r)
b_r <- train %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by  = "userId") %>%
  left_join(b_g, by = "genre") %>%
  left_join(b_y, by = "year") %>%
  group_by(review) %>%
  summarize(b_r = mean(rating - mu - b_i - b_u - b_g - b_y))

# RMSE based on mu, b_i, b_u, b_g, b_y and b_r effect
predicted_b_r <- test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genre") %>%
  left_join(b_y, by = "year") %>%
  left_join(b_r, by = "review") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y + b_r) %>%
  pull(pred)

RMSE(test$rating, predicted_b_r)

# 10 highest rated movies before regularization in the edx dataset
edx_set %>% group_by(movieId, title) %>%
  summarize(avg_rating = mean(rating), 
            n_ratings = n()) %>%
  arrange(desc(avg_rating))

# Finding the best lambda
# Try values 0 - 10 in increment of 0.25 to determine the best lambda 
lambdas <- seq(from=0, to=10, by=0.25)

# Regularise the model, predicting ratings and calculating RMSE for each value of lambda in the range lambdas
rmses <- sapply(lambdas, function(l){
  
  # Calculate average rating across training data
  mu <- mean(train$rating)
  
  # Compute regularized movie bias term
  b_i <- train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # Compute regularized user bias term
  b_u <- train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # Compute regularized genre bias term
  b_g <- train %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genre) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/ (n() + l))
  
  # Compute regularized year bias term
  b_y <- train %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genre") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u - b_g) / (n() +l))
  
  # Compute regularized review year term
  b_r <- train %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genre") %>%
    left_join(b_y, by = "year") %>%
    group_by(review) %>%
    summarize(b_r = sum(rating - mu - b_i - b_u - b_g - b_y) / (n() + l))
  
  # Compute predictions on test set based on these above terms
  predicted_ratings <- test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genre") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_r, by = "review") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_y + b_r) %>%
    pull(pred)
  
  # Output RMSE of these predictions on the test set
  return(RMSE(predicted_ratings, test$rating))})

# Find best lambda 
lambda <- lambdas[which.min(rmses)]
lambda

# Lambdas vs RMSE
qplot(lambdas, rmses)

# 10 Highest rated movies after regularization in the edx set
edx_set %>%
  group_by(movieId, title) %>%
  summarize(n_ratings = n(),
    b_i = sum(rating - mu) / (n() + lambda),
    reg_rating = mu + b_i,
    .groups = "drop") %>%
  arrange(desc(reg_rating)) %>%
  slice_head(n = 10)

# Effect of regularization on test set 
mu <- mean(test$rating)
avgs <- test %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

reg_avgs <- test %>%
  group_by(movieId) %>%
  summarize(
    b_i = sum(rating - mu) / (n() + lambda),
    n_i = n())

tibble(original = avgs$b_i,
       regularlized = reg_avgs$b_i,
       n = reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) +
  geom_point(shape=1, alpha=0.5)

# "The effect of factorization"
# Movie ratings correlation - latent factors
  # Count number of ratings per movie and per user
movie_counts <- edx %>% count(movieId)
user_counts  <- edx %>% count(userId)

  # Filter edx set for movies with more than 600 ratings and user who rated more than 600 movies
train_small <- edx %>%
  semi_join(movie_counts %>% filter(n >= 600), by = "movieId") %>%
  semi_join(user_counts  %>% filter(n >= 600), by = "userId")

  # Title in the dataset
available_titles <- sort(unique(train_small$title))

  # Select popular movies for correlation example
movies_for_table <- c(
  "Godfather, The (1972)",
  "Godfather: Part II, The (1974)",
  "Goodfellas (1990)",
  "Pride & Prejudice (2005)",
  "Sense and Sensibility (1995)",
  "Casino (1995)")

  # Prepare wide matrix of ratings
ratings_wide <- train_small %>%
  filter(title %in% movies_for_table) %>%
  select(userId, title, rating) %>%
  pivot_wider(names_from = title, values_from = rating) %>%
  column_to_rownames(var = "userId")

  # Determine the degree of correlations 
cor_table <- cor(ratings_wide, use = "pairwise.complete.obs") %>%
  round(3) %>%
  as.data.frame()

library(knitr)
knitr::kable(cor_table)


## Final Models and Results

# Regularized bias model

# Using full edx dataset to model effect of bias, and regularize with the best lambda 

# With movie bias effect
b_i <- edx_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda))

# With user bias effect
b_u <- edx_set %>% 
  left_join(b_i, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))

# With genre bias effect
b_g <- edx_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genre) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u) / (n() + lambda))


# With year bias effect
b_y <- edx_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>% 
  left_join(b_g, by = "genre") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u - b_g) / (n() + lambda))

# With review bias effect
b_r <- edx_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genre") %>%
  left_join(b_y, by = "year") %>%
  group_by(review) %>%
  summarize(b_r = sum(rating - mu - b_i - b_u - b_g - b_y) / (n() + lambda))

# Predict ratings for validation set 
predicted_validation <- validation_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genre") %>%
  left_join(b_y, by = "year") %>%
  left_join(b_r, by = "review") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y + b_r) %>%
  pull(pred)

# Calculate best RMSE for validation set
RMSE(validation_set$rating, predicted_validation)

# The effect of regularization
library(recosystem)

  # Set a reproducible seed
set.seed(1, sample.kind = "Rounding")

  # Convert train_set and validation_set to recosystem format
train_data <- with(edx, data_memory(
  user_index = userId,
  item_index = movieId,
  rating = rating))

test_data <- with(final_holdout_test, data_memory(
  user_index = userId,
  item_index = movieId,
  rating = rating))

  # Create the Recosystem model
r <- Reco()

  # Tune hyperparameters on train_set using cross-validation
opts <- r$tune(
  train_data,
  opts = list(
    dim = c(10, 20, 30),       # latent factor dimensions
    costp_l2 = c(0.01, 0.1),   # regularization for user factors
    costq_l2 = c(0.01, 0.1),   # regularization for item factors
    costp_l1 = 0, costq_l1 = 0,
    lrate = c(0.01, 0.1),      # learning rates
    nthread = 1,
    niter = 10,
    verbose = FALSE))

  # Train the model using the best tuning parameters
r$train(
  train_data,
  opts = c(opts$min, nthread = 1, niter = 100, verbose = FALSE))

  # Predict ratings for validation_set only
reco_pred <- r$predict(test_data, out_memory())

  # Calculate RMSE on validation_set
validation_rmse <- RMSE(final_holdout_test$rating, reco_pred)
validation_rmse

library(tibble)
library(knitr)

# RMSE 
RMSE_results <- tibble(
  Model = c(
    "RMSE Target",
    "Global Mean Model",
    "Regularized Bias Model",
    "Matrix Factorization (Recosystem)"
  ),
  RMSE = c(
    0.86490,
    1.060331,
    0.8643227,
    0.7818844))

kable(RMSE_results,
      digits = 6,
      caption = "Comparison of RMSE Across Recommendation Models")
