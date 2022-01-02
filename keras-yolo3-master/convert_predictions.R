## Read and create predictions to easily understood format

preds <- read.csv("/zhome/5b/5/109382/Desktop/DeepLearning/keras-yolo3-master/output/preds.txt")

library(dplyr)

sort(unique(preds$image)) ## 66 missing
## create 66!

preds <- rbind(preds, c("FRAME00066.jpg", 0, 0, 0, 0, 0, 0, 0))

preds <- within(
  preds,{
  image <- as.factor(image)
  box.xmin <- as.numeric(box.xmin)
  box.ymin <- as.numeric(box.ymin)
  box.xmax <- as.numeric(box.xmax)
  box.ymax <- as.numeric(box.ymax)
  Cyclist <- as.numeric(Cyclist)
  Helmet <- as.numeric(Helmet)
  Hovding <- as.numeric(Hovding)
}
)

summary <-
  preds %>%
  mutate(Bike = if_else(Cyclist >= 0.5,1,0)) %>%
  mutate(Helm = if_else(Helmet >= 0.5,1,0)) %>%
  mutate(Hovd = if_else(Hovding >= 0.5,1,0)) %>%
  group_by(image) %>%
  summarize(predBike = sum(Bike), predHelm = sum(Helm), predHovd = sum(Hovd))

write.csv(summary, "/zhome/5b/5/109382/Desktop/DeepLearning/keras-yolo3-master/output/formated_preds.csv", row.names = TRUE)

