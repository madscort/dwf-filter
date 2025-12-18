#!/usr/bin/env Rscript
#
# mads
# 2025-03-27
#
# Plot variant calling and filtering performance for pinpointables.

library(readr)
library(dplyr)
library(ggplot2)
library(forcats)
library(patchwork)

titlesize <- 8
textsize <- 7
subtitlesize <- 8

plot_data <- 
  read_tsv('data/variant_filter_benchmark_pins.tsv', show_col_types = F) %>% 
  filter(filtering != 'baseline') %>% 
  filter(filtering != 'HT') %>% 
  mutate(caller = if_else(caller_type == 'joint', 'GATK joint', 'GATK')) %>% 
  mutate(direction = if_else(((metric == 'F1' | metric == 'sensitivity') & (value - baseline > 0)) |
                               (metric == 'FDR' & (value - baseline < 0)), 'up', 'down')) %>% 
  mutate(filtering = factor(filtering, levels = c('HT', 'ML-S', 'ML-F1', 'VQSR')),
         caller_position = interaction(caller, filtering))

# Plotting function:
create_metric_plot <- function(data, variant_type_filter, metric_filter, y_limits, plot_title) {
  data %>% 
    filter(variant_type == variant_type_filter) %>% 
    filter(metric == metric_filter) %>% 
    ggplot(aes(x = caller_position, y = value)) +
    geom_point(aes(x = caller_position, y = baseline, color = "Baseline"), size = 0.7) +
    geom_segment(aes(y = baseline, yend = value, color = direction),
                 arrow = arrow(length = unit(0.02, "npc")),
                 size = 0.7,
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

# Plot all pinpointables:
sens_pin_all_plot <- create_metric_plot(plot_data, 'all', 'sensitivity', c(0.8, 1), 'Sensitivity')
fdr_pin_all_plot <- create_metric_plot(plot_data, 'all', 'FDR', c(0, 0.15), 'FDR')
f1_pin_all_plot <- create_metric_plot(plot_data, 'all', 'F1', c(0.8, 1), 'F1-score')
pin_bench_all_plot <- sens_pin_all_plot | fdr_pin_all_plot | f1_pin_all_plot + theme()

# SNVs
sens_pin_snv_plot <- create_metric_plot(plot_data, 'SNV', 'sensitivity', c(0.8, 1), 'Sensitivity')
fdr_pin_snv_plot <- create_metric_plot(plot_data, 'SNV', 'FDR', c(0, 0.15), 'FDR')
f1_pin_snv_plot <- create_metric_plot(plot_data, 'SNV', 'F1', c(0.8, 1), 'F1-score')

# INDELs
sens_pin_indel_plot <- create_metric_plot(plot_data, 'INDEL', 'sensitivity', c(0.0, 0.5), 'Sensitivity')
fdr_pin_indel_plot <- create_metric_plot(plot_data, 'INDEL', 'FDR', c(0, 0.5), 'FDR')
f1_pin_indel_plot <- create_metric_plot(plot_data, 'INDEL', 'F1', c(0.0, 1), 'F1-score')

# Combine SNVs into panel (a): 
sens_pin_snv_plot <- sens_pin_snv_plot + ggtitle("a - SNVs", subtitle = "Sensitivity") + 
  theme(plot.title = element_text(face = "bold", size = titlesize),
        plot.subtitle=element_text(size = subtitlesize))
fdr_pin_snv_plot <- fdr_pin_snv_plot + ggtitle("", subtitle = "FDR") + 
  theme(plot.title = element_text(face = "bold", size = titlesize),
        plot.subtitle=element_text(size = subtitlesize))
f1_pin_snv_plot <- f1_pin_snv_plot + ggtitle("", subtitle = "F1-score") + 
  theme(plot.title = element_text(face = "bold", size = titlesize),
        plot.subtitle=element_text(size = subtitlesize))

pin_bench_snv_plot <- sens_pin_snv_plot | fdr_pin_snv_plot | f1_pin_snv_plot

# Combine INDELs into panel (b): 
sens_pin_indel_plot <- sens_pin_indel_plot + ggtitle("b - Indels", subtitle = "Sensitivity") + 
  theme(plot.title = element_text(face = "bold", size = titlesize),
        plot.subtitle=element_text(size = subtitlesize))
fdr_pin_indel_plot <- fdr_pin_indel_plot + ggtitle("", subtitle = "FDR") + 
  theme(plot.title = element_text(face = "bold", size = titlesize),
        plot.subtitle=element_text(size = subtitlesize))
f1_pin_indel_plot <- f1_pin_indel_plot + ggtitle("", subtitle = "F1-score") + 
  theme(plot.title = element_text(face = "bold", size = titlesize),
        plot.subtitle=element_text(size = subtitlesize))

pin_bench_indel_plot <- sens_pin_indel_plot | fdr_pin_indel_plot | f1_pin_indel_plot

# Combine both panels into final plot:
pin_bench_plot <- pin_bench_snv_plot / pin_bench_indel_plot

ggsave('variant_filter_benchmark_pins.png',
       pin_bench_plot,
       dpi=600,
       width = 178,
       height = 118.66,
       units = "mm")