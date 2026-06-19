###script for Figures 3, 4 and Extended Data Figure 3
setwd('/Volumes/yuan_lab/TIER2/artemis_lei/codes/TMESegBiopsy_public/figure')
library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(corrplot)
library(RColorBrewer)
library(caret)
library(car) 
library(pROC) 
library(pheatmap)


#Figure 3b
data <- read.csv('../data/artemis_discovery.csv')
data_sub <- data[c('sTIL', 'AI_sTILratio', 'AI_sTILdensity', 'AI_TILratio', 'AI_TILdensity')]
M<-cor(data_sub, method='spearman')
#p-val
cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}
# matrix of the p-value of the correlation
p.mat <- cor.mtest(data_sub, method = "spearman", exact = FALSE)
# Benjamini-Hochberg (BH) adjustment
p.adjusted <- p.mat
p.adjusted[upper.tri(p.mat)] <- p.adjust(p.mat[upper.tri(p.mat)], method = "BH")
p.adjusted[lower.tri(p.mat)] <- t(p.adjusted)[lower.tri(p.mat)]  # Make it symmetric
#P-val adjusted
#sTIL vs AI-sTILratio: rho=0.733, P=6.364625e-35
#sTIL vs AI-sTILdensity: rho=0.739, P=1.019033e-35
#sTIL vs AI-TILratio: rho=0.491, P=1.425671e-13
#sTIL vs AI-TILdensity: rho=0.654, P=6.593250e-26

pdf('./fig3b-circle-spearman.pdf', height = 4, width = 4,onefile = FALSE)
corrplot(
  M, 
  method = "circle", 
  type = 'upper', 
  order = "original",
  hclust.method = "complete",
  col =rev(brewer.pal(n = 10, name = "RdBu")), # Expanded color range
  is.corr = FALSE, # Set to FALSE since your values are not standard correlations (-1 to 1)
  tl.col = "black", 
  tl.srt = 45,
  p.mat = p.adjusted, 
  sig.level = 0.05, 
  diag = FALSE,
  col.lim = c(0, 1) # Adjusted for your value range
)
dev.off()


#Figure3c-e, AUC
data <- data %>%
  mutate(pCR.bin = ifelse(pCR == "Yes", 1, 0)) %>%
  mutate(Ki.67.cate = ifelse(Ki.67 >= 40, 'High', 'Low')) %>%
  mutate(sTIL.cate = ifelse(sTIL >=20, 'High', 'Low')) %>%
  mutate(AI_sTILratio.cate = ifelse(AI_sTILratio >= 0.484, "High", "Low")) %>% 
  mutate(AI_sTILdensity.cate =ifelse(AI_sTILdensity >=1951.00, "High", "Low")) %>%  
  mutate(AI_TILratio.cate =ifelse(AI_TILratio >=0.173, "High", "Low")) %>%  
  mutate(AI_TILdensity.cate =ifelse(AI_TILdensity >=1675.76, "High", "Low")) 

data[, 10:15] <- lapply(data[, 10:15], factor, levels = c("Low", "High"))


# Figure3c
variables_MVA <- c("Ki.67.cate", "sTIL.cate", "AI_sTILratio.cate", "AI_sTILdensity.cate", "AI_TILratio.cate", "AI_TILdensity.cate") 
dep_var <- "pCR.bin" 

in_df_nona <- data
for(var in c(variables_MVA, dep_var)) {
  in_df_nona <- in_df_nona[!is.na(in_df_nona[[var]]),]
}

###important: need to manual change the number in variables_MVA[c(1,2,6)]
f <- as.formula(paste(dep_var,
                      paste(variables_MVA[c(1,2,6)], collapse = " + "),
                      sep = " ~ "))
res_glm <- glm(f , data = in_df_nona, family = "binomial")
summary_glm <- summary(res_glm)

odds_ratio <- as.data.frame(exp(Confint(res_glm))[-1,]) %>%  #remove intercept
  rownames_to_column() %>%
  dplyr::rename(predictor = rowname)  

coef_vec <- coef(summary(res_glm))[,1][-1]
coef_pval_vec <- coef(summary(res_glm))[,4][-1]

res_df <- odds_ratio %>%
  mutate(coef_pval = coef_pval_vec)
write.csv(res_df, paste0('MVA_', variables_MVA[1], '_', variables_MVA[2], '_', variables_MVA[6],'.csv'), row.names = FALSE)



#AUC,Pre and ACC for continuous TIL variables
continuous_var <- c('sTIL', 'AI_sTILratio', 'AI_sTILdensity', 'AI_TILratio', 'AI_TILdensity') 
categorical_var <- c("sTIL.cate", 'AI_sTILratio.cate', 'AI_sTILdensity.cate', 'AI_TILratio.cate', 'AI_TILdensity.cate')
variables_AUC <- continuous_var
dep_var <- "pCR.bin" 

in_df_nona <- data
for(var in c(variables_AUC, dep_var)) {
  in_df_nona <- in_df_nona[!is.na(in_df_nona[[var]]),]
}

###important: need to manual change the number in variables_AUC[c(1,5)]
f <- as.formula(paste(dep_var,
                      paste(variables_AUC[c(1,5)], collapse = " + "),
                      sep = " ~ "))
res_glm0 <- glm(f , data = in_df_nona, family = "binomial") #sTIL, 1
res_glm1 <- glm(f , data = in_df_nona, family = "binomial") #AI_sTILratio, 2
res_glm2 <- glm(f , data = in_df_nona, family = "binomial") #AI_sTILdensity, 3
res_glm3 <- glm(f , data = in_df_nona, family = "binomial") #AI_TILratio, 4
res_glm4 <- glm(f , data = in_df_nona, family = "binomial") #AI_TILdensity, 5
res_glm5 <- glm(f , data = in_df_nona, family = "binomial") #sTIL + AI_sTILratio, 1,2
res_glm6 <- glm(f , data = in_df_nona, family = "binomial") #sTIL + AI_sTILdensity, 1,3
res_glm7 <- glm(f , data = in_df_nona, family = "binomial") #sTIL + AI_TILratio, 1,4
res_glm8 <- glm(f , data = in_df_nona, family = "binomial") #sTIL + AI_TILdensity, 1,5

#predicted probabilities
in_df_nona$prob0 <- predict(res_glm0, type = "response")
in_df_nona$prob1 <- predict(res_glm1, type = "response")
in_df_nona$prob2 <- predict(res_glm2, type = "response")
in_df_nona$prob3 <- predict(res_glm3, type = "response")
in_df_nona$prob4 <- predict(res_glm4, type = "response")
in_df_nona$prob5 <- predict(res_glm5, type = "response")
in_df_nona$prob6 <- predict(res_glm6, type = "response")
in_df_nona$prob7 <- predict(res_glm7, type = "response")
in_df_nona$prob8 <- predict(res_glm8, type = "response")


roc_obj0 <- roc(in_df_nona$pCR.bin, in_df_nona$prob0, plot=TRUE) #the number of DPs is subject to the unique values of predicted probs
roc_obj1 <- roc(in_df_nona$pCR.bin, in_df_nona$prob1, plot=TRUE) 
roc_obj2 <- roc(in_df_nona$pCR.bin, in_df_nona$prob2, plot=TRUE) 
roc_obj3 <- roc(in_df_nona$pCR.bin, in_df_nona$prob3, plot=TRUE) 
roc_obj4 <- roc(in_df_nona$pCR.bin, in_df_nona$prob4, plot=TRUE) 
roc_obj5 <- roc(in_df_nona$pCR.bin, in_df_nona$prob5, plot=TRUE) 
roc_obj6 <- roc(in_df_nona$pCR.bin, in_df_nona$prob6, plot=TRUE) 
roc_obj7 <- roc(in_df_nona$pCR.bin, in_df_nona$prob7, plot=TRUE) 
roc_obj8 <- roc(in_df_nona$pCR.bin, in_df_nona$prob8, plot=TRUE) 

#Figure3c
roc_list <- list(
  roc_obj0 = roc_obj0,
  roc_obj1 = roc_obj1,
  roc_obj2 = roc_obj2,
  roc_obj3 = roc_obj3,
  roc_obj4 = roc_obj4
)

roc_data <- imap_dfr(roc_list, function(roc_obj, model_name) {
  data.frame(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities,
    model = model_name
  )
}) %>%
  mutate(model = factor(model, levels = names(roc_list)))

fig3c <- ggplot(roc_data, aes(x = fpr, y = tpr, color = model)) +
  geom_line() +
  geom_abline(linetype = "dashed", color = "gray") +
  scale_color_manual(values = c( "roc_obj0" = "#CC79A7", "roc_obj1" = "#f2cc8f", "roc_obj2" = "#F8961E", 
                                 "roc_obj3" = "#A1D99B", "roc_obj4" = "#43AA8B"), na.value = "#bababa",  drop = F ) + 
  labs(x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_classic() +
  theme(legend.position = "none",
        axis.title.x = element_text(size = 10),
        axis.title.y = element_text(size = 10),
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size=10))+
  annotate("text", x = 0.7, y = 0.25, size = 1,label = sprintf("sTIL (AUC = %.3f)", auc(roc_obj0)), color = "#CC79A7") +
  annotate("text", x = 0.7, y = 0.2, size = 1, label = sprintf("AI-sTILratio (AUC = %.3f)", auc(roc_obj1)), color = "#f2cc8f") +
  annotate("text", x = 0.7, y = 0.15, size = 1, label = sprintf("AI-sTILdensity (AUC = %.3f)", auc(roc_obj2)), color = "#F8961E") +
  annotate("text", x = 0.7, y = 0.1, size = 1, label = sprintf("AI-TILratio (AUC = %.3f)", auc(roc_obj3)), color = "#A1D99B") +
  annotate("text", x = 0.7, y = 0.05, size = 1, label = sprintf("AI-TILdensity (AUC = %.3f)", auc(roc_obj4)), color = "#43AA8B") 

ggsave(fig3c, file = "fig3c_discovery_AUC_con.pdf",
       width = 6, height = 5, units = "cm")

#Figure3c-ACC and precision
in_df_nona$pred0 <- ifelse(in_df_nona$prob0 > 0.5, 1, 0)
cm0 <- confusionMatrix(factor(in_df_nona$pred0), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred1 <- ifelse(in_df_nona$prob1 > 0.5, 1, 0)
cm1 <- confusionMatrix(factor(in_df_nona$pred1), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred2 <- ifelse(in_df_nona$prob2 > 0.5, 1, 0)
cm2 <- confusionMatrix(factor(in_df_nona$pred2), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred3 <- ifelse(in_df_nona$prob3 > 0.5, 1, 0)
cm3 <- confusionMatrix(factor(in_df_nona$pred3), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred4 <- ifelse(in_df_nona$prob4 > 0.5, 1, 0)
cm4 <- confusionMatrix(factor(in_df_nona$pred4), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))

roc_list <- list(
  "sTIL"          = roc_obj0,
  "AI-sTILratio"  = roc_obj1,
  "AI-sTILdensity"= roc_obj2,
  "AI-TILratio"   = roc_obj3,
  "AI-TILdensity" = roc_obj4
)

cm_list <- list(
  "sTIL"          = cm0,
  "AI-sTILratio"  = cm1,
  "AI-sTILdensity"= cm2,
  "AI-TILratio"   = cm3,
  "AI-TILdensity" = cm4
)

stats_data <- imap_dfr(roc_list, function(roc_obj, model_name) {
  auc_ci <- ci.auc(roc_obj)  # returns: lower, AUC, upper
  data.frame(
    model = model_name,
    AUC = round(as.numeric(auc(roc_obj)), 3),
    AUC_lower95 = round(as.numeric(auc_ci[1]), 3),
    AUC_upper95 = round(as.numeric(auc_ci[3]), 3),
    Accuracy = round(cm_list[[model_name]]$overall["Accuracy"], 3),
    Precision = round(cm_list[[model_name]]$byClass["Precision"], 3)
  )
}) %>%
  mutate(model = factor(model, levels = names(roc_list)))
write.csv(stats_data, 'Figure3c_discovery_AUC_acc_pre_con.csv', row.names = FALSE)


#Extended Data Figure 3b,
roc_list <- list(
  roc_obj0 = roc_obj0,
  roc_obj5 = roc_obj5,
  roc_obj6 = roc_obj6,
  roc_obj7 = roc_obj7,
  roc_obj8 = roc_obj8
)

roc_data <- imap_dfr(roc_list, function(roc_obj, model_name) {
  data.frame(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities,
    model = model_name
  )
}) %>%
  mutate(model = factor(model, levels = names(roc_list)))

fig3c <- ggplot(roc_data, aes(x = fpr, y = tpr, color = model)) +
  geom_line() +
  geom_abline(linetype = "dashed", color = "gray") +
  scale_color_manual(values = c( "roc_obj0" = "#CC79A7", "roc_obj5" = "#f2cc8f", "roc_obj6" = "#F8961E", 
                                 "roc_obj7" = "#A1D99B", "roc_obj8" = "#43AA8B"), na.value = "#bababa",  drop = F ) + 
  labs(x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_classic() +
  theme(legend.position = "none",
        axis.title.x = element_text(size = 10),
        axis.title.y = element_text(size = 10),
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size=10))+
  annotate("text", x = 0.7, y = 0.25, size = 1,label = sprintf("sTIL (AUC = %.3f)", auc(roc_obj0)), color = "#CC79A7") +
  annotate("text", x = 0.7, y = 0.2, size = 1, label = sprintf("sTIL+AI-sTILratio (AUC = %.3f)", auc(roc_obj5)), color = "#f2cc8f") +
  annotate("text", x = 0.7, y = 0.15, size = 1, label = sprintf("sTIL+AI-sTILdensity (AUC = %.3f)", auc(roc_obj6)), color = "#F8961E") +
  annotate("text", x = 0.7, y = 0.1, size = 1, label = sprintf("sTIL+AI-TILratio (AUC = %.3f)", auc(roc_obj7)), color = "#A1D99B") +
  annotate("text", x = 0.7, y = 0.05, size = 1, label = sprintf("sTIL+AI-TILdensity (AUC = %.3f)", auc(roc_obj8)), color = "#43AA8B") 

ggsave(fig3c, file = "EDfig3b_discovery_AUC_con.pdf",
       width = 6, height = 5, units = "cm")

#Extended Data Figure3b-ACC and precision
in_df_nona$pred0 <- ifelse(in_df_nona$prob0 > 0.5, 1, 0)
cm0 <- confusionMatrix(factor(in_df_nona$pred0), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred5 <- ifelse(in_df_nona$prob5 > 0.5, 1, 0)
cm5 <- confusionMatrix(factor(in_df_nona$pred5), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred6 <- ifelse(in_df_nona$prob6 > 0.5, 1, 0)
cm6 <- confusionMatrix(factor(in_df_nona$pred6), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred7 <- ifelse(in_df_nona$prob7 > 0.5, 1, 0)
cm7 <- confusionMatrix(factor(in_df_nona$pred7), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred8 <- ifelse(in_df_nona$prob8 > 0.5, 1, 0)
cm8 <- confusionMatrix(factor(in_df_nona$pred8), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))

roc_list <- list(
  "sTIL"          = roc_obj0,
  "sTIL+AI-sTILratio"  = roc_obj5,
  "sTIL+AI-sTILdensity"= roc_obj6,
  "sTIL+AI-TILratio"   = roc_obj7,
  "sTIL+AI-TILdensity" = roc_obj8
)

cm_list <- list(
  "sTIL"          = cm0,
  "sTIL+AI-sTILratio"  = cm5,
  "sTIL+AI-sTILdensity"= cm6,
  "sTIL+AI-TILratio"   = cm7,
  "sTIL+AI-TILdensity" = cm8
)

stats_data <- imap_dfr(roc_list, function(roc_obj, model_name) {
  auc_ci <- ci.auc(roc_obj)  # returns: lower, AUC, upper
  data.frame(
    model = model_name,
    AUC = round(as.numeric(auc(roc_obj)), 3),
    AUC_lower95 = round(as.numeric(auc_ci[1]), 3),
    AUC_upper95 = round(as.numeric(auc_ci[3]), 3),
    Accuracy = round(cm_list[[model_name]]$overall["Accuracy"], 3),
    Precision = round(cm_list[[model_name]]$byClass["Precision"], 3)
  )
}) %>%
  mutate(model = factor(model, levels = names(roc_list)))
write.csv(stats_data, 'EDFigure3b_discovery_AUC_acc_pre_con.csv', row.names = FALSE)




#AUC,Pre and ACC for categorical TIL variables
#Extended Data Figure3d
categorical_var <- c("sTIL.cate", 'AI_sTILratio.cate', 'AI_sTILdensity.cate', 'AI_TILratio.cate', 'AI_TILdensity.cate')
variables_AUC <- categorical_var
dep_var <- "pCR.bin" 

in_df_nona <- data
for(var in c(variables_AUC, dep_var)) {
  in_df_nona <- in_df_nona[!is.na(in_df_nona[[var]]),]
}

###important: need to manual change the number in variables_AUC[c(1,5)]
f <- as.formula(paste(dep_var,
                      paste(variables_AUC[c(1,5)], collapse = " + "),
                      sep = " ~ "))
res_glm0 <- glm(f , data = in_df_nona, family = "binomial") #sTIL, 1
res_glm1 <- glm(f , data = in_df_nona, family = "binomial") #AI_sTILratio, 2
res_glm2 <- glm(f , data = in_df_nona, family = "binomial") #AI_sTILdensity, 3
res_glm3 <- glm(f , data = in_df_nona, family = "binomial") #AI_TILratio, 4
res_glm4 <- glm(f , data = in_df_nona, family = "binomial") #AI_TILdensity, 5
res_glm5 <- glm(f , data = in_df_nona, family = "binomial") #sTIL + AI_sTILratio, 1,2
res_glm6 <- glm(f , data = in_df_nona, family = "binomial") #sTIL + AI_sTILdensity, 1,3
res_glm7 <- glm(f , data = in_df_nona, family = "binomial") #sTIL + AI_TILratio, 1,4
res_glm8 <- glm(f , data = in_df_nona, family = "binomial") #sTIL + AI_TILdensity, 1,5

#predicted probabilities
in_df_nona$prob0 <- predict(res_glm0, type = "response")
in_df_nona$prob1 <- predict(res_glm1, type = "response")
in_df_nona$prob2 <- predict(res_glm2, type = "response")
in_df_nona$prob3 <- predict(res_glm3, type = "response")
in_df_nona$prob4 <- predict(res_glm4, type = "response")
in_df_nona$prob5 <- predict(res_glm5, type = "response")
in_df_nona$prob6 <- predict(res_glm6, type = "response")
in_df_nona$prob7 <- predict(res_glm7, type = "response")
in_df_nona$prob8 <- predict(res_glm8, type = "response")

roc_obj0 <- roc(in_df_nona$pCR.bin, in_df_nona$prob0, plot=TRUE) #the number of DPs is subject to the unique values of predicted probs
roc_obj1 <- roc(in_df_nona$pCR.bin, in_df_nona$prob1, plot=TRUE) 
roc_obj2 <- roc(in_df_nona$pCR.bin, in_df_nona$prob2, plot=TRUE) 
roc_obj3 <- roc(in_df_nona$pCR.bin, in_df_nona$prob3, plot=TRUE) 
roc_obj4 <- roc(in_df_nona$pCR.bin, in_df_nona$prob4, plot=TRUE) 
roc_obj5 <- roc(in_df_nona$pCR.bin, in_df_nona$prob5, plot=TRUE) 
roc_obj6 <- roc(in_df_nona$pCR.bin, in_df_nona$prob6, plot=TRUE) 
roc_obj7 <- roc(in_df_nona$pCR.bin, in_df_nona$prob7, plot=TRUE) 
roc_obj8 <- roc(in_df_nona$pCR.bin, in_df_nona$prob8, plot=TRUE) 

in_df_nona$pred0 <- ifelse(in_df_nona$prob0 > 0.5, 1, 0)
cm0 <- confusionMatrix(factor(in_df_nona$pred0), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred1 <- ifelse(in_df_nona$prob1 > 0.5, 1, 0)
cm1 <- confusionMatrix(factor(in_df_nona$pred1), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred2 <- ifelse(in_df_nona$prob2 > 0.5, 1, 0)
cm2 <- confusionMatrix(factor(in_df_nona$pred2), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred3 <- ifelse(in_df_nona$prob3 > 0.5, 1, 0)
cm3 <- confusionMatrix(factor(in_df_nona$pred3), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred4 <- ifelse(in_df_nona$prob4 > 0.5, 1, 0)
cm4 <- confusionMatrix(factor(in_df_nona$pred4), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred5 <- ifelse(in_df_nona$prob5 > 0.5, 1, 0)
cm5 <- confusionMatrix(factor(in_df_nona$pred5), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred6 <- ifelse(in_df_nona$prob6 > 0.5, 1, 0)
cm6 <- confusionMatrix(factor(in_df_nona$pred6), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred7 <- ifelse(in_df_nona$prob7 > 0.5, 1, 0)
cm7 <- confusionMatrix(factor(in_df_nona$pred7), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred8 <- ifelse(in_df_nona$prob8 > 0.5, 1, 0)
cm8 <- confusionMatrix(factor(in_df_nona$pred8), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))

roc_list <- list(
  "sTIL"          = roc_obj0,
  "AI-sTILratio"  = roc_obj1,
  "AI-sTILdensity"= roc_obj2,
  "AI-TILratio"   = roc_obj3,
  "AI-TILdensity" = roc_obj4,
  "sTIL+AI-sTILratio"  = roc_obj5,
  "sTIL+AI-sTILdensity"= roc_obj6,
  "sTIL+AI-TILratio"   = roc_obj7,
  "sTIL+AI-TILdensity" = roc_obj8
)

cm_list <- list(
  "sTIL"          = cm0,
  "AI-sTILratio"  = cm1,
  "AI-sTILdensity"= cm2,
  "AI-TILratio"   = cm3,
  "AI-TILdensity" = cm4,
  "sTIL+AI-sTILratio"  = cm5,
  "sTIL+AI-sTILdensity"= cm6,
  "sTIL+AI-TILratio"   = cm7,
  "sTIL+AI-TILdensity" = cm8
)

stats_data <- imap_dfr(roc_list, function(roc_obj, model_name) {
  auc_ci <- ci.auc(roc_obj)  # returns: lower, AUC, upper
  data.frame(
    model = model_name,
    AUC = round(as.numeric(auc(roc_obj)), 3),
    AUC_lower95 = round(as.numeric(auc_ci[1]), 3),
    AUC_upper95 = round(as.numeric(auc_ci[3]), 3),
    Accuracy = round(cm_list[[model_name]]$overall["Accuracy"], 3),
    Precision = round(cm_list[[model_name]]$byClass["Precision"], 3)
  )
}) %>%
  mutate(model = factor(model, levels = names(roc_list)))
write.csv(stats_data, 'EDFigure3d_discovery_AUC_acc_pre_cate.csv', row.names = FALSE)


##Figure 3e
#categorical variables for AUC plot, Ki-67 + sTIL (obj10), Ki-67 + count16 (obj11), Ki-67 + sTIL +count16 (obj12)
in_df_nona <- data
for(var in c(variables_AUC, dep_var)) {
  in_df_nona <- in_df_nona[!is.na(in_df_nona[[var]]),]
}

f10 <- as.formula("pCR.bin ~ Ki.67.cate + sTIL.cate")
f11 <- as.formula("pCR.bin ~ Ki.67.cate + AI_sTILdensity.cate")
f12 <- as.formula("pCR.bin ~ Ki.67.cate + sTIL.cate + AI_sTILdensity.cate")

res_glm10 <- glm(f10, data = in_df_nona, family = binomial)
res_glm11 <- glm(f11, data = in_df_nona, family = binomial)
res_glm12 <- glm(f12, data = in_df_nona, family = binomial)

roc_obj10 <- roc(in_df_nona$pCR.bin, fitted(res_glm10))
roc_obj11 <- roc(in_df_nona$pCR.bin, fitted(res_glm11))
roc_obj12 <- roc(in_df_nona$pCR.bin, fitted(res_glm12))

roc_list <- list(
  "Ki-67 + sTILs" = roc_obj10,
  "Ki-67 + AI-sTILdensity" = roc_obj11,
  "Ki-67 + sTILs + AI-sTILdensity" = roc_obj12
)

model_colors <- c(
  "Ki-67 + sTILs" = "#CC79A7",
  "Ki-67 + AI-sTILdensity" = "#F8961E",
  "Ki-67 + sTILs + AI-sTILdensity" = "#0072B2"
)

roc_data <- imap_dfr(roc_list, function(roc_obj, model_name) {
  data.frame(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities,
    model = model_name
  )
}) %>%
  mutate(model = factor(model, levels = names(roc_list)))

auc_labels <- imap_dfr(roc_list, function(roc_obj, model_name) {
  data.frame(
    model = model_name,
    label = sprintf("%s (AUC = %.3f)", model_name, auc(roc_obj)),
    x = 0.10,
    y = c(0.25, 0.15, 0.05)[match(model_name, names(roc_list))]
  )
}) %>%
  mutate(model = factor(model, levels = names(roc_list)))

fig3e <- ggplot(roc_data, aes(x = fpr, y = tpr, color = model)) +
  geom_line(linewidth = 0.6) +
  geom_abline(linetype = "dashed", color = "gray") +
  geom_text(
    data = auc_labels,
    aes(x = x, y = y, label = label, color = model),
    size = 1.8,
    hjust = 0,
    inherit.aes = FALSE
  ) +
  scale_color_manual(values = model_colors,na.value = "#bababa",drop = FALSE) +
  labs(
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  coord_equal() +
  theme_classic() +
  theme(
    legend.position = "none",
    axis.title.x = element_text(size = 6),
    axis.title.y = element_text(size = 6),
    axis.text.x = element_text(size = 6),
    axis.text.y = element_text(size = 6)
  )

ggsave(filename = "fig3e_discovery_AUC_cate.pdf",plot = fig3e,width = 6,height = 5,units = "cm")

###Figure 3e - ACC,precision
in_df_nona$prob10 <- predict(res_glm10, type = "response")
in_df_nona$prob11 <- predict(res_glm11, type = "response")
in_df_nona$prob12 <- predict(res_glm12, type = "response")

in_df_nona$pred10 <- ifelse(in_df_nona$prob10 > 0.5, 1, 0)
cm10 <- confusionMatrix(factor(in_df_nona$pred10), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred11 <- ifelse(in_df_nona$prob11 > 0.5, 1, 0)
cm11 <- confusionMatrix(factor(in_df_nona$pred11), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))
in_df_nona$pred12 <- ifelse(in_df_nona$prob12 > 0.5, 1, 0)
cm12 <- confusionMatrix(factor(in_df_nona$pred12), factor(in_df_nona$pCR.bin), positive='1',dnn = c("Prediction", "Reference"))

roc_list <- list(
  "Ki-67 + sTILs"          = roc_obj10,
  "Ki-67 + AI-sTILdensity"  = roc_obj11,
  "Ki-67 + sTILs + AI-sTILdensity"= roc_obj12
)

cm_list <- list(
  "Ki-67 + sTILs"  = cm10,
  "Ki-67 + AI-sTILdensity"  = cm11,
  "Ki-67 + sTILs + AI-sTILdensity"= cm12
)

stats_data <- imap_dfr(roc_list, function(roc_obj, model_name) {
  auc_ci <- ci.auc(roc_obj)  # returns: lower, AUC, upper
  data.frame(
    model = model_name,
    AUC = round(as.numeric(auc(roc_obj)), 3),
    AUC_lower95 = round(as.numeric(auc_ci[1]), 3),
    AUC_upper95 = round(as.numeric(auc_ci[3]), 3),
    Accuracy = round(cm_list[[model_name]]$overall["Accuracy"], 3),
    Precision = round(cm_list[[model_name]]$byClass["Precision"], 3)
  )
}) %>%
  mutate(model = factor(model, levels = names(roc_list)))
write.csv(stats_data, 'Figure3e_discovery_AUC_acc_pre_cate.csv', row.names = FALSE)



##Extended Data Figure 3
### need to download the supp data of PMID: 29628290
immune <- read_xlsx('../data/Immunity2018_1-s2.0-S1074761318301213-mmc2.xlsx')
immune <- immune %>%
  filter(`TCGA Study` == 'BRCA')
colnames(immune) <- gsub(" ", ".", colnames(immune))
colnames(immune)[colnames(immune)=='TCGA.Participant.Barcode'] <- 'patient_id'
write.csv(immune, '../data/Immunity2018_BRCA.csv', row.names = FALSE)

#TCGA TCBC call was based on the supp data of PMID: 34725325
til <- read.csv( '../data/TCGA-TNBC_AIscore.csv', )
immune <- read.csv('../data/Immunity2018_BRCA.csv')
til_immune <- til %>%
  left_join(immune, by='patient_id')

til_immune_set1 <- til_immune %>%
  dplyr::select(1:5, 9, 10, 41:62) 

colnames(til_immune_set1)
# [1] "patient_id"                   "AI_sTILratio"                 "AI_sTILdensity"               "AI_TILratio"                 
#[5] "AI_TILdensity"                "Leukocyte.Fraction"           "Stromal.Fraction"             "B.Cells.Memory"              
#[9] "B.Cells.Naive"                "Dendritic.Cells.Activated"    "Dendritic.Cells.Resting"      "Eosinophils...41"            
#[13] "Macrophages.M0"               "Macrophages.M1"               "Macrophages.M2"               "Mast.Cells.Activated"        
#[17] "Mast.Cells.Resting"           "Monocytes"                    "Neutrophils...48"             "NK.Cells.Activated"          
#[21] "NK.Cells.Resting"             "Plasma.Cells"                 "T.Cells.CD4.Memory.Activated" "T.Cells.CD4.Memory.Resting"  
#[25] "T.Cells.CD4.Naive"            "T.Cells.CD8"                  "T.Cells.Follicular.Helper"    "T.Cells.gamma.delta"         
#[29] "T.Cells.Regulatory.Tregs" 

# to achieve the absolute percentage
til_immune_set2 <- til_immune_set1 %>%
  mutate(
    across(
      8:ncol(til_immune_set1),
      ~ .x * Leukocyte.Fraction
    )
  ) %>%
  filter(Leukocyte.Fraction>=0)

immune_long <- til_immune_set2 %>%
  dplyr::select(8:ncol(til_immune_set2), AI_sTILratio, AI_sTILdensity, AI_TILratio, AI_TILdensity) %>%
  pivot_longer(
    cols = 1:22,
    names_to = "cell_type",
    values_to = "cell_fraction"
  )

cor_results <- immune_long %>%
  pivot_longer(
    cols = c(AI_sTILratio, AI_sTILdensity, AI_TILratio, AI_TILdensity),
    names_to = "variable",
    values_to = "variable_value"
  ) %>%
  group_by(cell_type, variable) %>%
  dplyr::summarize(
    spearman_rho = cor(cell_fraction, variable_value,
                       method = "spearman",
                       use = "pairwise.complete.obs"),
    p_value = cor.test(cell_fraction, variable_value,
                       method = "spearman", exact = FALSE)$p.value,
    .groups = "drop"
  )

cor_results <- cor_results %>%
  group_by(variable) %>%
  mutate(
    p_adj_BH = p.adjust(p_value, method = "BH")
  ) %>%
  ungroup()


###corr heatmap
vars <- c( "AI_sTILratio", "AI_sTILdensity", "AI_TILratio", "AI_TILdensity")

cor_mat <- map_dfc(
  vars,
  ~ sapply(
    til_immune_set2[, 8:ncol(til_immune_set2)],
    function(x)
      cor(x, til_immune_set2[[.x]],
          method = "spearman",
          use = "pairwise.complete.obs")
  )
)

rownames(cor_mat) <- colnames(til_immune_set2)[8:ncol(til_immune_set2)]
colnames(cor_mat) <- vars



pheatmap(
  cor_mat,
  color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  border_color = NA,
  fontsize_row = 9,
  fontsize_col = 10,
  main = "Spearman correlation between immune cell types and sTIL metrics"
)

p_mat <- map_dfc(
  vars,
  ~ sapply(
    til_immune_set2[, 8:ncol(til_immune_set2)],
    function(x)
      cor.test(x, til_immune_set2[[.x]], method = "spearman")$p.value
  )
)
rownames(p_mat) <- rownames(cor_mat)
colnames(p_mat) <- vars

p_adj <- apply(p_mat, 2, p.adjust, method = "BH")
sig_labels <- matrix("", nrow = nrow(p_adj), ncol = ncol(p_adj))
sig_labels[p_adj < 0.05] <- "*"
sig_labels[p_adj < 0.01] <- "**"
sig_labels[p_adj < 0.001] <- "***"
sig_labels[p_adj < 0.0001] <- "****"

pdf("TCGA_TNBC_correlation_heatmap.pdf", width = 5, height = 5)
pheatmap(
  cor_mat,
  color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
  breaks = seq(-0.5, 0.5, length.out = 101),
  display_numbers = sig_labels,
  number_color = "black",
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  border_color = NA,
  fontsize_row = 9,
  fontsize_col = 10
)
dev.off()
