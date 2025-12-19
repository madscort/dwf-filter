#!/usr/bin/env Rscript
#
# mads
# 2025-12-17
#
# Plot variant calling and filtering performance

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(forcats)
library(patchwork)

titlesize <- 8
textsize <- 7
subtitlesize <- 8

# Calculate metrics from TP/FP/FN
calc_metrics <- function(tp, fp, fn) {
  tibble(
    TP = tp,
    FP = fp,
    FN = fn,
    sensitivity = tp / (tp + fn),
    precision = tp / (tp + fp),
    F1 = (2 * tp) / (2 * tp + fp + fn),
    FDR = fp / (fp + tp)
  )
}

pool_df <-
  read_tsv('data/variant_filter_benchmark_pools.tsv', show_col_types = F) %>% 
  mutate(filter_method = if_else(caller=='mlsens','ML-S',
                                 if_else(caller=='mlf1','ML-F1',
                                         if_else(caller=='vqsr','VQSR',
                                                 if_else(caller=='ht','HT',
                                                         if_else(caller=='single','baseline',
                                                                 if_else(caller=='joint','baseline',caller)))))),
         caller_type = if_else(caller=='mlsens' | caller=='mlf1' | caller=='ht' | caller=='single','single','joint')) %>% 
  rename(config = caller)

# Aggregate by config and variant_type
aggregated <- pool_df %>% 
  group_by(config,filter_method, caller_type, variant_type) %>%
  summarise(TP = sum(TP), FP = sum(FP), FN = sum(FN), .groups = 'drop') %>%
  rowwise() %>%
  mutate(calc_metrics(TP, FP, FN)) %>%
  ungroup()

# Get baseline metrics
baseline <- aggregated %>%
  filter(filter_method == 'baseline') %>%
  select(caller_type, variant_type, 
         baseline_sensitivity = sensitivity,
         baseline_F1 = F1,
         baseline_FDR = FDR)

# Join baseline to filtering methods and convert to long format
plot_data <- aggregated %>%
  filter(filter_method != 'baseline') %>%
  left_join(baseline, by = c('caller_type', 'variant_type')) %>%
  pivot_longer(cols = c(sensitivity, F1, FDR),
               names_to = 'metric',
               values_to = 'value') %>%
  mutate(baseline = case_when(
    metric == 'sensitivity' ~ baseline_sensitivity,
    metric == 'F1' ~ baseline_F1,
    metric == 'FDR' ~ baseline_FDR
  )) %>%
  select(filtering = filter_method, caller_type, variant_type, metric, value, baseline) %>%
  mutate(variant_type = if_else(variant_type == 'BOTH', 'all', variant_type)) %>%
  mutate(caller = if_else(caller_type == 'joint', 'GATK joint', 'GATK')) %>%
  mutate(direction = if_else(((metric == 'F1' | metric == 'sensitivity') & (value - baseline > 0)) |
                               (metric == 'FDR' & (value - baseline < 0)), 'up', 'down')) %>%
  mutate(filtering = factor(filtering, levels = c('HT', 'ML-S', 'ML-F1', 'VQSR')),
         caller_position = interaction(caller, filtering))

create_metric_plot <- function(data, variant_type_filter, metric_filter, y_limits, plot_title) {
  data %>% 
    filter(variant_type == variant_type_filter) %>% 
    filter(metric == metric_filter) %>% 
    ggplot(aes(x = caller_position, y = value)) +
    geom_point(aes(x = caller_position, y = baseline, color = "Baseline"), size = 0.7) +
    geom_segment(aes(y = baseline, yend = value, color = direction),
                 arrow = arrow(length = unit(0.02, "npc")),
                 linewidth = 0.7,
                 show.legend = FALSE) +
    scale_color_manual(name = NULL,
                       values = c("up" = "darkgreen",
                                  "down" = "darkred",
                                  "Baseline" = "black"),
                       breaks = c("Baseline")) +
    scale_x_discrete(labels = function(x) sub("^.*\\.", "", x)) +
    theme_bw() +
    theme(axis.title = element_text(size = textsize),
          axis.text = element_text(size = textsize),
          plot.subtitle=element_text(size = subtitlesize),
          plot.title=element_text(size = titlesize),
          legend.position = '') +
    ylim(y_limits[1], y_limits[2]) +
    labs(x = NULL, y = NULL) +
    ggtitle(plot_title)
}

# Plot all variants:
sens_pin_all_plot <- create_metric_plot(plot_data, 'all', 'sensitivity', c(0.5, 1.0), 'Sensitivity')
fdr_pin_all_plot <- create_metric_plot(plot_data, 'all', 'FDR', c(0.0, 0.5), 'FDR')
f1_pin_all_plot <- create_metric_plot(plot_data, 'all', 'F1', c(0.5, 1.0), 'F1-score')
bench_plot <- sens_pin_all_plot | fdr_pin_all_plot | f1_pin_all_plot

ggsave('./variant_filter_benchmark_pool.png',
       bench_plot,
       dpi=600,
       width = 178,
       height = 118.66/2,
       units = "mm")