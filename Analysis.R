rm(list = ls())
library(psych)
library(car)
library(ggplot2)
library(tidyverse)
library(pwr)
library(multcomp)
library(outliers)
library(FSA)
library(gtable)
library(gridExtra)

#sixhump
sixhump.df<-read.csv('sd1.5/sixhumpsd15.csv', header = T)
sixhump.df <- dplyr::select(sixhump.df,-X)
sixhump.df$algorithm <- as.factor(sixhump.df$algorithm)
sixhump.df$function. <- as.factor(sixhump.df$function.)

#easom
easom.df<-read.csv('sd1.5/easomsd15.csv', header = T)
easom.df <- dplyr::select(easom.df,-X)
easom.df$algorithm <- as.factor(easom.df$algorithm)
easom.df$function. <- as.factor(easom.df$function.)
#miele
miele.df<-read.csv('sd1.5/mielesd15 2.csv', header = T)
miele.df <- dplyr::select(miele.df,-X)
miele.df$algorithm <- as.factor(miele.df$algorithm)
miele.df$function. <- as.factor(miele.df$function.)
#paviani
paviani.df<-read.csv('sd1.5/pavianisd15.csv', header = T)
paviani.df <- dplyr::select(paviani.df,-X)
paviani.df$algorithm <- as.factor(paviani.df$algorithm)
paviani.df$function. <- as.factor(paviani.df$function.)

#total for plotting
df <- rbind(sixhump.df,easom.df,miele.df,paviani.df)
df$true_reward_difference[df$true_reward_difference>1e6] <- NA
df$euclidean_distance[df$euclidean_distance>1e6] <- NA
df$cumulative_regret[df$cumulative_regret>1e6] <- NA
df$cumulative_regret[df$timetocomple>1e6] <- NA


#Box-plot
boxplot_euclidean <- ggplot(df, aes(x=algorithm, y=euclidean_distance)) +
  geom_boxplot()+
  facet_wrap(~function., scale='free')+
  theme_light()+
  labs(y="Euclidean distance", 
       x = "Algorithm", 
       title='Minimum Euclidean distance of the maximum points')
boxplot_euclidean
# ggsave('boxplot_euclidean.png', 
#        device = 'png',
#        scale = 2,
#        width = 8,
#        height = 8,
#        units = 'cm',
#        dpi = 300)


boxplot_reward <- ggplot(df, aes(x=algorithm, y=true_reward_difference)) +
  geom_boxplot()+
  facet_wrap(~function., scale='free')+
  theme_light()+
  labs(y="Reward difference", 
     x = "Algorithm", 
     title='Reward difference with the maximum reward')
boxplot_reward
# ggsave('boxplot_reward.png', 
#        device = 'png',
#        scale = 2,
#        width = 8,
#        height = 8,
#        units = 'cm',
#        dpi = 300)

boxplot_regret <- ggplot(df, aes(x=algorithm, y=cumulative_regret)) +
  geom_boxplot()+
  facet_wrap(~function., scale='free')+
  theme_light()+
  labs(y="Cumulative Regret", 
       x = "Algorithm", 
       title='Cumulative regret for the optimization')
boxplot_regret
# ggsave('boxplot_regret.png', 
#        device = 'png',
#        scale = 2,
#        width = 8,
#        height = 8,
#        units = 'cm',
#        dpi = 300)

boxplot_time <- ggplot(df, aes(x=algorithm, y=timetocomple)) +
  geom_boxplot()+
  facet_wrap(~function., scale='free')+
  theme_light()+
  labs(y="Time (s)", 
       x = "Algorithm", 
       title='Time to complete optimization in the horizon')
boxplot_time
# ggsave('boxplot_time.png', 
#        device = 'png',
#        scale = 2,
#        width = 8,
#        height = 8,
#        units = 'cm',
#        dpi = 300)

grid.arrange(boxplot_euclidean, boxplot_reward, boxplot_regret, boxplot_time, nrow = 2)



#Descriptive statistics
describeBy(sixhump.df$euclidean_distance, sixhump.df$algorithm, mat =T, digits = 3)
describeBy(sixhump.df$true_reward_difference, sixhump.df$algorithm, mat =T, digits = 3)
describeBy(sixhump.df$cumulative_regret, sixhump.df$algorithm, mat =T, digits = 3)
describeBy(sixhump.df$timetocomple, sixhump.df$algorithm, mat =T, digits = 3)

describeBy(easom.df$euclidean_distance, easom.df$algorithm, mat =T, digits = 3)
describeBy(easom.df$true_reward_difference, easom.df$algorithm, mat =T, digits = 3)
describeBy(easom.df$cumulative_regret, easom.df$algorithm, mat =T, digits = 3)
describeBy(easom.df$timetocomple, easom.df$algorithm, mat =T, digits = 3)

describeBy(miele.df$euclidean_distance, miele.df$algorithm, mat =T, digits = 3)
describeBy(miele.df$true_reward_difference, miele.df$algorithm, mat =T, digits = 3)
describeBy(miele.df$cumulative_regret, miele.df$algorithm, mat =T, digits = 3)
describeBy(miele.df$timetocomple, miele.df$algorithm, mat =T, digits = 3)

describeBy(paviani.df$euclidean_distance, paviani.df$algorithm, mat =T, digits = 3)
describeBy(paviani.df$true_reward_difference, paviani.df$algorithm, mat =T, digits = 3)
describeBy(paviani.df$cumulative_regret, paviani.df$algorithm, mat =T, digits = 3)
describeBy(paviani.df$timetocomple, paviani.df$algorithm, mat =T, digits = 3)

#Cum regret
#Kruskal-Wallis test for 
kruskal.test(cumulative_regret ~ algorithm, data=sixhump.df)
kruskal.test(cumulative_regret ~ algorithm, data=easom.df)
kruskal.test(cumulative_regret ~ algorithm, data=miele.df)
kruskal.test(cumulative_regret ~ algorithm, data=paviani.df)
#Multiple comparisons with Dunn test
dunnTest(cumulative_regret ~ algorithm, data=sixhump.df)
dunnTest(cumulative_regret ~ algorithm, data=easom.df)
dunnTest(cumulative_regret ~ algorithm, data=miele.df)
dunnTest(cumulative_regret ~ algorithm, data=paviani.df)

#Euclidean distance
#Kruskal-Wallis test for 
kruskal.test(euclidean_distance ~ algorithm, data=sixhump.df)
kruskal.test(euclidean_distance ~ algorithm, data=easom.df)
kruskal.test(euclidean_distance ~ algorithm, data=miele.df)
kruskal.test(euclidean_distance ~ algorithm, data=paviani.df)
#Multiple comparisons with Dunn test
dunnTest(euclidean_distance ~ algorithm, data=sixhump.df)
dunnTest(euclidean_distance ~ algorithm, data=easom.df)
?dunnTest(euclidean_distance ~ algorithm, data=miele.df)
dunnTest(euclidean_distance ~ algorithm, data=paviani.df)

#Rewad difference distance
#Kruskal-Wallis test for 
kruskal.test(true_reward_difference ~ algorithm, data=sixhump.df)
kruskal.test(true_reward_difference ~ algorithm, data=easom.df)
kruskal.test(true_reward_difference ~ algorithm, data=miele.df)
kruskal.test(true_reward_difference ~ algorithm, data=paviani.df)
#Multiple comparisons with Dunn test
dunnTest(true_reward_difference ~ algorithm, data=sixhump.df)
dunnTest(true_reward_difference ~ algorithm, data=easom.df)
dunnTest(true_reward_difference ~ algorithm, data=miele.df)
dunnTest(true_reward_difference ~ algorithm, data=paviani.df)

#Time
#Kruskal-Wallis test
kruskal.test(timetocomple ~ algorithm, data=sixhump.df)
kruskal.test(timetocomple ~ algorithm, data=easom.df)
kruskal.test(timetocomple ~ algorithm, data=miele.df)
kruskal.test(timetocomple ~ algorithm, data=paviani.df)
#Multiple comparisons with Dunn test
dunnTest(timetocomple ~ algorithm, data=sixhump.df)
dunnTest(timetocomple ~ algorithm, data=easom.df)
dunnTest(timetocomple ~ algorithm, data=miele.df)
dunnTest(timetocomple ~ algorithm, data=paviani.df)