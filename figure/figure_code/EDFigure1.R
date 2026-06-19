library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(purrr)
library(RColorBrewer)

fold_dir <- "../data"

# ----- load all folds -----
df <- map_dfr(1:5, \(k) {
  read_csv(file.path(fold_dir, paste0("test_patch_metrics_fold", k, ".csv")),
           show_col_types = FALSE) %>%
    mutate(fold = factor(k))
})

# ----- reshape Dice + NPV to long -----
dice_long <- df %>%
  dplyr::select(fold, file_name, starts_with("dice_"), -matches("^dice_mean")) %>%
  pivot_longer(
    cols = starts_with("dice_"),
    names_to = "class",
    values_to = "value"
  ) %>%
  mutate(
    metric = "Dice",
    # class like "dice_1_tumor" -> "tumor"
    class = sub("^dice_\\d+_", "", class)
  )

npv_long <- df %>%
  dplyr::select(fold, file_name, starts_with("npv_"), -matches("^npv_macro")) %>%
  pivot_longer(
    cols = starts_with("npv_"),
    names_to = "class",
    values_to = "value"
  ) %>%
  mutate(
    metric = "NPV",
    class = sub("^npv_\\d+_", "", class)
  )

recall_long <- df %>%
  dplyr::select(fold, file_name, starts_with("recall_")) %>%
  pivot_longer(
    cols = starts_with("recall_"),
    names_to = "class",
    values_to = "value"
  ) %>%
  mutate(
    metric = "Recall",
    class = sub("^recall_\\d+_", "", class)
  )


long <- bind_rows(dice_long, npv_long, recall_long) %>%
  # drop background if present
  filter(class != "background") %>%
  mutate(
    # nice class ordering (edit to your desired order)
    class = factor(class, levels = c("tumor", "stroma", "benign_epi", "necrosis_hemo", "adipose"))
  )

# ----- summarize within fold (mean ± SE over patches) -----
sum_df <- long %>%
  group_by(metric, class, fold) %>%
  summarise(
    mean = mean(value, na.rm = TRUE),
    se   = sd(value, na.rm = TRUE) / sqrt(sum(!is.na(value))),
    ci   = qt(0.975, df = sum(!is.na(value)) - 1) * se,
    .groups = "drop"
  )

sum_df_dice <- sum_df %>%
  filter(metric == "Dice") %>%
  group_by(class) %>%
  summarise(mean = mean(mean, na.rm = TRUE))

sum_df_npv <- sum_df %>%
  filter(metric == "NPV") %>%
  group_by(class) %>%
  summarise(mean = mean(mean, na.rm = TRUE))

sum_df_recall <- sum_df %>%
  filter(metric == "Recall") %>%
  group_by(class) %>%
  summarise(mean = mean(mean, na.rm = TRUE))


# ----- plot (facet by metric: Dice and NPV) -----
#Figure2b-dice
paired_odd <- brewer.pal(12, "Paired")[c(1, 3, 5, 7, 9)]
p_dice <- sum_df %>% filter(metric == "Dice") %>%
  ggplot(aes(class, mean, fill = fold)) +
  geom_col(position = position_dodge(width = 0.85), width = 0.75) +
  geom_errorbar(
    aes(ymin = mean - ci, ymax = mean + ci),
    position = position_dodge(width = 0.85),
    width = 0.3
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = NULL, y = NULL, fill = "Fold") +
  scale_fill_manual(values = paired_odd) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(), # Remove minor grid lines
    axis.line.y = element_line(color = "black"), # Add y-axis line
    axis.line.x = element_line(color = "black"), # Add y-axis line
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
print(p_dice)

# Save
ggsave(p_dice, file='fig2b_5CV_dice.pdf', width =9, height = 6, units = "cm", dpi = 300)

#Extended Figure1c-npv
p_npv <- sum_df %>% filter(metric == "NPV") %>%
  ggplot(aes(class, mean, fill = fold)) +
  geom_col(position = position_dodge(width = 0.85), width = 0.75) +
  geom_errorbar(
    aes(ymin = mean - ci, ymax = mean + ci),
    position = position_dodge(width = 0.85),
    width = 0.3
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = NULL, y = NULL, fill = "Fold") +
  scale_fill_manual(values = paired_odd) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(), # Remove minor grid lines
    axis.line.y = element_line(color = "black"), # Add y-axis line
    axis.line.x = element_line(color = "black"), # Add y-axis line
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
print(p_npv)
ggsave(p_npv, file='EDfig1c_5CV_npv.pdf', width =9, height = 6, units = "cm", dpi = 300)

