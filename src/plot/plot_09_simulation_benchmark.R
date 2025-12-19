#!/usr/bin/env Rscript
#
# mads
# 2025-03-27
#
# Plot variant calling and filtering performance for pinpointables.

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(forcats)
library(patchwork)

titlesize <- 8
textsize <- 7
subtitlesize <- 8

df_pool <- 
  read_tsv('data/simulation_benchmark.tsv', show_col_types = F)

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


aggregated_pool <- 
  df_pool %>% 
  group_by(matrix_size,coverage,filtration_status,variant_type) %>%
  summarise(TP = sum(TP), FP = sum(FP), FN = sum(FN), .groups = 'drop') %>%
  rowwise() %>%
  mutate(calc_metrics(TP, FP, FN)) %>%
  ungroup()

pool_plot_data <-
  aggregated_pool %>% 
  mutate(matrix_size = factor(paste0(matrix_size,'x',matrix_size), levels= c('2x2','5x5','10x10','24x24'))) %>% 
  mutate(coverage = factor(paste0(coverage,'x'), levels = c('5x','10x','30x','100x'))) %>% 
  select(-c('TP','FP','FN')) %>% 
  pivot_longer(cols = c('sensitivity','precision','F1','FDR'),
               values_to = 'value', names_to = 'metric') %>% 
  pivot_wider(names_from = filtration_status, values_from = value, names_prefix = 'value_')


plot_metrics_data <- function(data, var_type, perf_metric, y_limits, plot_title) {
  data %>% 
    filter(variant_type == var_type) %>% 
    filter(metric == perf_metric) %>% 
    ggplot(aes(x=factor(matrix_size),
               y=value_unfiltered,
               colour=factor(coverage),
               group = factor(coverage))) +
    geom_point(position = position_dodge(width = 0.5), size = 0.85) +
    geom_segment(aes(y = y_limits[1], 
                     yend = value_unfiltered, 
                     color = factor(coverage)),
                 #arrow = arrow(length = unit(0.02, "npc")),
                 size = 0.6,
                 show.legend = FALSE,
                 position = position_dodge(width = 0.5)) +
    theme_minimal() +
    theme(legend.title = element_blank(),
          axis.title = element_blank(),
          axis.text = element_text(size = textsize),
          plot.subtitle=element_text(size = subtitlesize),
          plot.title=element_text(size = titlesize)) +
    ylab(perf_metric) +
    ylim(y_limits[1], y_limits[2]) +
    ggthemes::scale_color_tableau() +
    ggtitle(plot_title)
}


# SNVs
sens_pool_snv_plot <- plot_metrics_data(pool_plot_data, 'SNV', 'sensitivity', c(0.0, 1), 'Sensitivity')
fdr_pool_snv_plot <- plot_metrics_data(pool_plot_data, 'SNV', 'FDR', c(0, 0.5), 'FDR')
f1_pool_snv_plot <- plot_metrics_data(pool_plot_data, 'SNV', 'F1', c(0.0, 1), 'F1-score')

# INDELs
sens_pool_indel_plot <- plot_metrics_data(pool_plot_data, 'INDEL', 'sensitivity', c(0.0, 1.0), 'Sensitivity')
fdr_pool_indel_plot <- plot_metrics_data(pool_plot_data, 'INDEL', 'FDR', c(0, 0.5), 'FDR')
f1_pool_indel_plot <- plot_metrics_data(pool_plot_data, 'INDEL', 'F1', c(0.0, 1.0), 'F1-score')

# Combine SNVs into panel (a): 
sens_pool_snv_plot <- sens_pool_snv_plot + ggtitle("a - SNVs", subtitle = "Sensitivity") + 
  theme(plot.title = element_text(face = "bold", size = titlesize),
        plot.subtitle=element_text(size = subtitlesize))
fdr_pool_snv_plot <- fdr_pool_snv_plot + ggtitle("", subtitle = "FDR") + 
  theme(plot.title = element_text(face = "bold", size = titlesize),
        plot.subtitle=element_text(size = subtitlesize))
f1_pool_snv_plot <- f1_pool_snv_plot + ggtitle("", subtitle = "F1-score") + 
  theme(plot.title = element_text(face = "bold", size = titlesize),
        plot.subtitle=element_text(size = subtitlesize))

pool_bench_snv_plot <- sens_pool_snv_plot | fdr_pool_snv_plot | f1_pool_snv_plot

# Combine INDELs into panel (b): 
sens_pool_indel_plot <- sens_pool_indel_plot + ggtitle("b - Indels", subtitle = "Sensitivity") + 
  theme(plot.title = element_text(face = "bold", size = titlesize),
        plot.subtitle=element_text(size = subtitlesize))
fdr_pool_indel_plot <- fdr_pool_indel_plot + ggtitle("", subtitle = "FDR") + 
  theme(plot.title = element_text(face = "bold", size = titlesize),
        plot.subtitle=element_text(size = subtitlesize))
f1_pool_indel_plot <- f1_pool_indel_plot + ggtitle("", subtitle = "F1-score") + 
  theme(plot.title = element_text(face = "bold", size = titlesize),
        plot.subtitle=element_text(size = subtitlesize))

pool_bench_indel_plot <- sens_pool_indel_plot | fdr_pool_indel_plot | f1_pool_indel_plot

# Combine both panels into final plot:
pool_bench_plot <- (pool_bench_snv_plot / pool_bench_indel_plot) + 
  plot_layout(guides = "collect") & theme(legend.position = "bottom",
                                          legend.margin = margin(t = -5, unit = "pt"),
                                          legend.text = element_text(size = textsize))

ggsave('simulation_benchmark.png',
       pool_bench_plot,
       dpi=600,
       width = 178,
       height = 118.66,
       units = "mm")