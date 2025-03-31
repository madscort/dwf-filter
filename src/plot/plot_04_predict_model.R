#!/usr/bin/env Rscript
#
# mads
# 2025-03-27
#
# Plot filter model performance

library(readr)
library(dplyr)
library(ggplot2)
library(forcats)
library(pROC)
library(boot)
library(patchwork)
library(ggpubr)

# Load model held-out prediction data:
predictions_snv <-
  read_tsv('data/heldout_predictions_snv.tsv', show_col_types = F)
predictions_indel <-
  read_tsv('data/heldout_predictions_indel.tsv', show_col_types = F)

# Function for calculating performance metrics during bootstrapping:
calc_metrics <- function(data, indices) {
  # Get bootstrap sample
  d <- data[indices,]
  
  # Calculate metrics
  auc <- auc(roc(d$y_true, d$y_proba, quiet=T))
  tp <- sum(d$y_true == 1 & d$y_pred == 1)
  fp <- sum(d$y_true == 0 & d$y_pred == 1) 
  fn <- sum(d$y_true == 1 & d$y_pred == 0)
  fdr <- fp/(tp + fp)
  sens <- tp/(tp + fn)
  prec <- tp/(tp + fp)
  f1 <- (2 * prec * sens) / (prec + sens)
  
  return(c(f1, auc, fdr, sens))
}

# Common plotting font size
common_theme <- theme(
  axis.title = element_text(size=10),
  axis.text = element_text(size=10),
  legend.text = element_text(size=10),
  legend.position = ""
)

# Bootstrap SNV predictions for each model
print('Bootstrapping SNV predictions. This takes a (long) while.')
results_snv <- predictions_snv %>%
  group_by(model_name) %>%
  do({
    boot_res <- boot(., calc_metrics, R=1000)
    data.frame(
      metric = c("F1-score", "ROC-AUC", "FDR", "Sensitivity"),
      mean = colMeans(boot_res$t),
      ci_lower = apply(boot_res$t, 2, quantile, 0.025),
      ci_upper = apply(boot_res$t, 2, quantile, 0.975)
    )
  })

# Bootstrap INDEL predictions for each model
print('Bootstrapping INDEL predictions. This takes a (short) while.')
results_indel <- predictions_indel %>%
  group_by(model_name) %>%
  do({
    boot_res <- boot(., calc_metrics, R=1000)
    data.frame(
      metric = c("F1-score", "ROC-AUC", "FDR", "Sensitivity"),
      mean = colMeans(boot_res$t),
      ci_lower = apply(boot_res$t, 2, quantile, 0.025),
      ci_upper = apply(boot_res$t, 2, quantile, 0.975)
    )
  })

# Create ROC curves for each model - SNV predictions
roc_curves_snv <- list()
for (model in unique(predictions_snv$model_name)) {
  filtered_data <- 
    predictions_snv %>%
    filter(model_name == model)
  roc_curve <- roc(filtered_data$y_true, filtered_data$y_proba, quiet = TRUE)
  roc_curves_snv[[model]] <- roc_curve
}

# Create ROC curves for each model - INDEL predictions
roc_curves_indel <- list()
for (model in unique(predictions_indel$model_name)) {
  filtered_data <- 
    predictions_indel %>%
    filter(model_name == model)
  roc_curve <- roc(filtered_data$y_true, filtered_data$y_proba, quiet = TRUE)
  roc_curves_indel[[model]] <- roc_curve
}
colors <- c("gmm" = "#009E73", "logistic_regression" = "#f1c40f", "random_forest" = "#0072B2", 'xgboost' = "#D55E00")

# Plot bootstrapping results SNV
metrics_snv <-
  results_snv %>% 
  mutate(model_name = factor(model_name, levels = sort(unique(model_name)))) %>% 
  ggplot(aes(x=metric, y=mean, fill=model_name)) +
  geom_bar(stat="identity", position=position_dodge(0.9)) +
  geom_errorbar(aes(ymin=ci_lower, ymax=ci_upper),
                width=0.2,
                position=position_dodge(0.9)) +
  scale_fill_manual(values = colors, name="Model") +
  theme_bw()  +
  xlab('') +
  ylab('Metric Value') +
  ylim(0,1)

# Plot bootstrapping results INDEL
metrics_indel <-
  results_indel %>% 
  mutate(model_name = factor(model_name, levels = c('gmm','logistic_regression','random_forest','xgboost'))) %>% 
  arrange(model_name) %>% 
  ggplot(aes(x=metric, y=mean, fill=model_name)) +
  geom_bar(stat="identity", position=position_dodge(0.9)) +
  geom_errorbar(aes(ymin=ci_lower, ymax=ci_upper),
                width=0.2,
                position=position_dodge(0.9)) +
  scale_fill_manual(values = colors, name="Model") +
  theme_bw() +
  theme(axis.title.y = element_blank()) +
  theme_bw()  +
  xlab('') +
  ylab('Metric Value') +
  ylim(0,1)


# Plot ROC SNV
roc_snv <- 
  roc_curves_snv %>% 
  ggroc(aes = c("color"), legacy.axes = TRUE) +
  scale_color_manual(values = colors, name = 'Model') +
  geom_abline(lty = 3) +
  labs(x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_equal() +
  theme(legend.position = '',
        axis.text = element_text(size=10)) +
  theme_bw() + 
  common_theme

# Plot ROC INDEL
roc_indel <-
  roc_curves_indel %>% 
  ggroc(aes = c("color"), legacy.axes = TRUE) +
  scale_color_manual(values = colors, name = 'Model') +
  geom_abline(lty = 3) +
  labs(x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_equal() +
  theme(legend.position = '',
        axis.text = element_text(size=10)) +
  theme_bw() + 
  common_theme

# Create final combined plot:

# Create clean labels for legend
model_levels <- levels(factor(results_snv$model_name))
clean_labels <- stringr::str_to_title(stringr::str_replace_all(model_levels, "_", " ")) %>% 
  stringr::str_replace_all("Gmm", "GMM") %>% 
  stringr::str_replace_all("Xgboost", "XGBoost")

# Create separate legend
legend_data <- data.frame(x = seq_along(model_levels),  # Different x-values
                          y = 1,  # Keep y constant
                          model_name = model_levels)
color_legend <- ggplot(legend_data, aes(x, y, color = model_name)) +
  geom_point(size = 5) +
  scale_color_manual(values = colors, name = "Model", labels = clean_labels) +
  theme_void() +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 12))
color_legend <- ggpubr::as_ggplot(ggpubr::get_legend(color_legend))

# Remove legend from plots
metrics_snv <- metrics_snv + theme(legend.position = "none") + ggtitle('b')
metrics_indel <- metrics_indel + theme(legend.position = "none") + ggtitle('d')
roc_snv <- roc_snv + theme(legend.position = "none") + ggtitle('a')
roc_indel <- roc_indel + theme(legend.position = "none") + ggtitle('c')

# Combined plots into figure
metrics_plot <- ((plot_spacer()/plot_spacer()) | (roc_snv / roc_indel) | 
                   (plot_spacer()/plot_spacer()) | (metrics_snv / metrics_indel)) +
  plot_layout(widths = c(-1.4, 4, -1.4, 2),
              heights = c(1, 1, 1, 1)) &
  common_theme

# Add separate legend
annotated_plot <- wrap_plots(metrics_plot, color_legend, ncol = 1, heights = c(10, 1))

ggsave('model_performance.png',
       annotated_plot,
       width = 10,
       height = 7.5)