
# CLearing Memory and setting Working Directory
rm(list=ls())
setwd('E:/Data Science(Edwisor)/Project/Project #1/NLP+DataSet+-+Sarcasm+Detection/NLP DataSet - Sarcasm Detection')

#install.packages('tidytext')
#Loading Libraries
library(jsonlite)
library(stringr)
library(tm)
library(wordcloud)
library(caTools)
library(randomForest)
library(e1071)
library(xgboost)
library(topicmodels)
library(tidytext)

#Loading the Data
json_data = stream_in(file("Sarcasm_Headlines_Dataset (1).json",encoding = 'utf-8'))

dataset = json_data[1:10000,1:2]
#Cleaning Data

dataset$headline = as.character(dataset$headline)

corpus = VCorpus(VectorSource(dataset$headline))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

#WORD CLOUD in the working directory
postCorpus_WC = corpus
pal2 = brewer.pal(8,"Dark2")
png("wordcloud_v2.png", width = 12, height = 8, units = 'in', res = 300)
wordcloud(postCorpus_WC, scale = c(5,.2), min.freq = 30, max.words = 200, random.order = FALSE, rot.per = .15, colors = pal2)
dev.off()




#CREATING TFIDF Matrix for the corpus
dtm = DocumentTermMatrix(corpus)
#dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))

dataset$is_sarcastic = json_data$is_sarcastic[1:10000]


#top 50 Sarcastic words
v = sort(rowSums(dataset[-11920]), decreasing=TRUE)
head(v, 50)
data_col = colnames(dataset)
data_col[]

#LDA for topic modelling
ap_lda <- LDA(dataset[-11920], k = 6, control = list(seed = 1234))
#ap_lda
ap_documents <- tidy(ap_lda, matrix = "gamma")
#View(ap_documents)

#Splitting Dataset into test and train
gc()
set.seed(123)
train_index = sample(1:nrow(dataset), 0.8 * nrow(dataset))
training_set = dataset[train_index,]
test_set = dataset[-train_index,]

ncol(dataset)
# Fitting Logistic Regression to the Training set
classifier = glm(formula = is_sarcastic ~ .,
                 family = binomial,
                 data = training_set)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-11920])
# Making the Confusion Matrix
cm = table(test_set[, 11920], y_pred)



#Fitting Naivye Bayes to the training set
classifier = naiveBayes(x = training_set[-11920],
                        y = training_set[11920])
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-11920])
# Making the Confusion Matrix
cm = table(test_set[, 11920], y_pred)




#Fitting Random Forest to the Training set
classifier = randomForest(x = training_set[-11920],
                          y = training_set[11920],
                          ntree = 100)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-11920])
# Making the Confusion Matrix
cm = table(test_set[, 11920], y_pred)




#Fitting XGBoost to the Training set
classifier = xgboost(data = training_set[-11920],label = training_set[11920])
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-11920])
# Making the Confusion Matrix
cm = table(test_set[, 11920], y_pred)



#Using DEEP learning for the further process
install.packages("kerasR")
#devtools::install_github("rstudio/keras")
#install_keras(method = "conda")
#install_keras(tensorflow = "gpu")
#tensorflow::install_tensorflow(restart_session = TRUE,conda="C:/Program Files/Python36/Scripts")

library(tensorflow)
library(keras)
library(kerasR)
#corpus = VCorpus(VectorSource(dataset$headline))
#apply(dataset$headline ,1, content_transformer(tolower))
#apply(dataset$headline ,1, removeNumbers)

dataset = json_data[,1:2]
train_index = sample(1:nrow(dataset), 0.9 * nrow(dataset))
training = dataset[train_index,]
testing = dataset[-train_index,]

num_words <- 15000
max_length <- 120
text_vectorization = layer_text_vectorization(max_tokens = num_words, output_sequence_length = max_length)

text_vectorization %>%  adapt(dataset$headline)


text_vectorization(matrix(dataset$headline[1],ncol =1))


input = layer_input(shape = c(1), dtype = "string")

output <- input %>% 
  text_vectorization() %>% 
  layer_embedding(input_dim = num_words + 1, output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")


model = keras_model(input, output)
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)


history <- model %>% fit(
  training$headline ,
  training$is_sarcastic ,
  epochs = 20,
  batch_size = 512,
  validation_split = 0.1,
  verbose=2
)


results <- model %>% evaluate(testing$headline ,testing$is_sarcastic , verbose = 0)
results
