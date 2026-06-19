###script for Figure 2 and Extended Data Figure 2
setwd('/Volumes/yuan_lab/TIER2/artemis_lei/codes/TMESegBiopsy_public/figure')
library(dplyr)
library(tidyr)
library(purrr)
library(tibble)
library(ggpubr) 
library(ggplot2)
library(caret) 

#Figure 2c, Extended Figure 2a-e
cell_eval <- read.csv('../data/cell_class_eval.csv')
cell_levels <- c("t", "l", "f", "o")
actual <- factor(cell_eval$class_annotation, levels = cell_levels)
predicted_raw <- factor(cell_eval$class_raw, levels = cell_levels)
predicted_refine <- factor(cell_eval$class_refine, levels = cell_levels)
predicted_hover <- factor(cell_eval$class_hover_match, levels = cell_levels)
predicted_cellvit <- factor(cell_eval$class_cellvit_match, levels = cell_levels)

#performance of raw
raw_actual_cm=confusionMatrix(predicted_raw, actual)
raw_actual = raw_actual_cm$byClass
raw_actual = data.frame(t(raw_actual))
colnames(raw_actual) = c("Tumor", "Lymphocytes", "Fibroblasts","Others")
raw_actual$Average = (raw_actual$Tumor + raw_actual$Lymphocytes + raw_actual$Fibroblasts)/3

#performance of refine
refine_actual_cm=confusionMatrix(predicted_refine, actual)
refine_actual = refine_actual_cm$byClass
refine_actual = data.frame(t(refine_actual))
colnames(refine_actual) = c("Tumor", "Lymphocytes", "Fibroblasts","Others")
refine_actual$Average = (refine_actual$Tumor + refine_actual$Lymphocytes + refine_actual$Fibroblasts)/3

#performance of hovernet
hover_actual_cm=confusionMatrix(predicted_hover, actual)
hover_actual = hover_actual_cm$byClass
hover_actual = data.frame(t(hover_actual))
colnames(hover_actual) = c("Tumor", "Lymphocytes", "Fibroblasts","Others")
hover_actual$Average = (hover_actual$Tumor + hover_actual$Lymphocytes + hover_actual$Fibroblasts)/3

#performance of cellvit++
cellvit_actual_cm=confusionMatrix(predicted_cellvit, actual)
cellvit_actual = cellvit_actual_cm$byClass
cellvit_actual = data.frame(t(cellvit_actual))
colnames(cellvit_actual) = c("Tumor", "Lymphocytes", "Fibroblasts","Others")
cellvit_actual$Average = (cellvit_actual$Tumor + cellvit_actual$Lymphocytes + cellvit_actual$Fibroblasts)/3


####plot CM heatmap
###refine vs. annotation (truth, rows)
t(refine_actual_cm$table)
cm_refine <- matrix(
  c(
    2444, 106,  29, 236,
    77, 2245, 95, 237,
    5,   99, 1398, 148,
    0,    0,    0,   0
  ),
  nrow = 4,
  byrow = TRUE
)

####raw vs. annotation (truth, rows)
t(raw_actual_cm$table)
cm_raw <- matrix(
  c(
    2685,   75,    3,   52,
    1060, 1477,   38,   79,
    478,   23, 1087,   62,
    0,    0,    0,   0
  ),
  nrow = 4,
  byrow = TRUE
)

####raw vs. refine (truth, rows)
raw_refine <- confusionMatrix(predicted_refine, predicted_raw)
raw_refine$table
cm_raw_refine <- matrix(
  c(
    2526,    0,    0,    0,
    900,  1550,    0,    0,
    394,    0,   1128,    0,
    403,   25,    0,  193
  ),
  nrow = 4,
  byrow = TRUE
)


#hovernet vs. annotation
t(hover_actual_cm$table)
cm_hover <- matrix(
  c(
    2659,   22,   23,  111,
    300,  2120,  144,   90,
    133,    9, 1435,   73,
    0,    0,    0,    0
  ),
  nrow = 4,
  byrow = TRUE
)

#cellvit vs. annotation
t(cellvit_actual_cm$table)
cm_cellvit <- matrix(
  c(
    2742,   33,   31,    9,
    824, 1613,  207,   10,
    203,   14, 1432,    1,
    0,    0,    0,    0
  ),
  nrow = 4,
  byrow = TRUE
)

color_map <- c(
  cm_refine = "#e07a5f",
  cm_raw = "#0072B2",
  cm_raw_refine = "#CC79A7",
  cm_hover = "#009E73",
  cm_cellvit = "#D6616B"
)

cm <- cm_cellvit
cm_color <- color_map['cm_cellvit']

cm_level <- c("Tumor", "Lymphocytes", "Fibroblasts","Others")
rownames(cm) <- cm_level
colnames(cm) <- cm_level

cm_df <- as.data.frame(cm) %>%
  rownames_to_column("True") %>%
  pivot_longer(
    cols = -True,
    names_to = "Predicted",
    values_to = "Count"
  ) %>%
  mutate(
    True = factor(True, levels = cm_level),
    Predicted = factor(Predicted, levels = cm_level)
  )

p_cm <- ggplot(cm_df, aes(x = Predicted, y = True, fill = Count)) +
  geom_tile(color = "gray", linewidth = 0.6) +
  geom_text(aes(label = Count), size = 4) +
  scale_fill_gradient(low = "white", high = cm_color) + 
scale_y_discrete(limits = rev(levels(cm_df$True))) +  # ← key line
  labs(
    x = "Predicted label",
    y = "True label",
    fill = "Count"
  ) +
  coord_fixed() +
  theme_minimal(base_size = 16) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
print(p_cm)
ggsave(p_cm, file = "CM-cell-cellvit.pdf",
       width = 10, height = 10, units = "cm") 



#Figure2d, cell classification performance
cell_eval <- read.csv('../data/cell_class_eval.csv')
class_levels <- c("t", "l", "f", "o")

f1_one_class <- function(actual, pred, cls) {
  actual <- factor(actual, levels = class_levels)
  pred   <- factor(pred,   levels = class_levels)
  
  TP <- sum(actual == cls & pred == cls, na.rm = TRUE)
  FP <- sum(actual != cls & pred == cls, na.rm = TRUE)
  FN <- sum(actual == cls & pred != cls, na.rm = TRUE)
  
  denom <- 2 * TP + FP + FN
  
  if (denom == 0) {
    return(NA_real_)
  } else {
    return(2 * TP / denom)
  }
}

f1_by_file <- cell_eval %>%
  group_by(file_source) %>%
  dplyr::summarise(
    t_F1_refine = f1_one_class(class_annotation, class_refine, "t"),
    l_F1_refine = f1_one_class(class_annotation, class_refine, "l"),
    f_F1_refine = f1_one_class(class_annotation, class_refine, "f"),
    o_F1_refine = f1_one_class(class_annotation, class_refine, "o"),
    t_F1_raw = f1_one_class(class_annotation, class_raw, "t"),
    l_F1_raw = f1_one_class(class_annotation, class_raw, "l"),
    f_F1_raw = f1_one_class(class_annotation, class_raw, "f"),
    o_F1_raw = f1_one_class(class_annotation, class_raw, "o"),
    .groups = "drop"
  )

f1_by_file$average_F1_refine <- (f1_by_file$t_F1_refine + f1_by_file$l_F1_refine + f1_by_file$f_F1_refine)/3
f1_by_file$average_F1_raw <- (f1_by_file$t_F1_raw + f1_by_file$l_F1_raw + f1_by_file$f_F1_raw)/3


#tumor cells
df_long <- f1_by_file %>%
  pivot_longer(cols = c(t_F1_raw, t_F1_refine), names_to = "Type", values_to = "F1_Score")

# Create paired boxplot
fig2d_t <- ggplot(df_long, aes(x = Type, y = F1_Score,  color = Type)) +
  geom_boxplot() +
  geom_point(position = position_dodge(width = 0.5), size = 1) +
  geom_line(aes(group = file_source), color = "black") +
  scale_color_manual(values = c("t_F1_raw" = "#BDBDBD", "t_F1_refine" = "seagreen3")) +
  stat_compare_means(paired = TRUE, method = "wilcox.test")+
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(), # Remove major grid lines
    panel.grid.minor = element_blank(), # Remove minor grid lines
    axis.ticks = element_line(linewidth = 0.5),
    axis.line.y = element_line(color = "black"), # Add y-axis line
    axis.line.x = element_line(color = "black"), # Add y-axis line
  )+
  theme(legend.position = "none") +
  ylim(c(0,1))+
  labs(x = "", y = "F1 Score")
print(fig2d_t)
ggsave(fig2d_t, file = "fig2d-tumor.pdf",
       width = 6, height = 6, units = "cm")


#lymphocytes
df_long <- f1_by_file %>%
  pivot_longer(cols = c(l_F1_raw, l_F1_refine), names_to = "Type", values_to = "F1_Score")

# Create paired boxplot
fig2d_l <- ggplot(df_long, aes(x = Type, y = F1_Score,  color = Type)) +
  geom_boxplot() +
  geom_point(position = position_dodge(width = 0.5), size = 1) +
  geom_line(aes(group = file_source), color = "black") +
  scale_color_manual(values = c("l_F1_raw" = "#BDBDBD", "l_F1_refine" = "blue")) +
  stat_compare_means(paired=TRUE, method = "wilcox.test")+
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(), # Remove major grid lines
    panel.grid.minor = element_blank(), # Remove minor grid lines
    axis.ticks = element_line(linewidth = 0.5),
    axis.line.y = element_line(color = "black"), # Add y-axis line
    axis.line.x = element_line(color = "black"), # Add y-axis line
  )+
  theme(legend.position = "none") +
  ylim(c(0,1))+
  labs(x = "", y = "F1 Score")
print(fig2d_l)
ggsave(fig2d_l, file = "fig2d-lymphocyte.pdf",
       width = 6, height = 6, units = "cm")

#fibroblasts
df_long <- f1_by_file %>%
  pivot_longer(cols = c(f_F1_raw, f_F1_refine), names_to = "Type", values_to = "F1_Score")

# Create paired boxplot
fig2d_f <- ggplot(df_long, aes(x = Type, y = F1_Score,  color = Type)) +
  geom_boxplot() +
  geom_point(position = position_dodge(width = 0.5), size = 1) +
  geom_line(aes(group = file_source), color = "black") +
  scale_color_manual(values = c("f_F1_raw" = "#BDBDBD", "f_F1_refine" = "#F0E442")) +
  stat_compare_means(paired = TRUE, method = "wilcox.test")+
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(), # Remove major grid lines
    panel.grid.minor = element_blank(), # Remove minor grid lines
    axis.ticks = element_line(linewidth = 0.5),
    axis.line.y = element_line(color = "black"), # Add y-axis line
    axis.line.x = element_line(color = "black"), # Add y-axis line
  )+
  theme(legend.position = "none") +
  ylim(c(0,1))+
  labs(x = "", y = "F1 Score")
print(fig2d_f)
ggsave(fig2d_f, file = "fig2d-fibroblast.pdf",
       width = 6, height = 6, units = "cm")

#average
df_long <- f1_by_file %>%
  pivot_longer(cols = c(average_F1_raw, average_F1_refine), names_to = "Type", values_to = "F1_Score")

# Create paired boxplot
fig2d_avg <- ggplot(df_long, aes(x = Type, y = F1_Score,  color = Type)) +
  geom_boxplot() +
  geom_point(position = position_dodge(width = 0.5), size = 1) +
  geom_line(aes(group = file_source), color = "black") +
  scale_color_manual(values = c("average_F1_raw" = "#BDBDBD", "average_F1_refine" = "#9E9AC8")) +
  stat_compare_means(paired = TRUE, method = "wilcox.test")+
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(), # Remove major grid lines
    panel.grid.minor = element_blank(), # Remove minor grid lines
    axis.ticks = element_line(linewidth = 0.5),
    axis.line.y = element_line(color = "black"), # Add y-axis line
    axis.line.x = element_line(color = "black"), # Add y-axis line
  )+
  theme(legend.position = "none") +
  ylim(c(0,1))+
  labs(x = "", y = "F1 Score")
print(fig2d_avg)
ggsave(fig2d_avg, file = "fig2d-avgerage.pdf",
       width = 6, height = 6, units = "cm")
