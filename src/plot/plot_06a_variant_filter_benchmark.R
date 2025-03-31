#!/usr/bin/env Rscript
#
# mads
# 2025-03-27
#
# Plot variant calling and filtering performance

library(readr)
library(dplyr)
library(ggplot2)
library(forcats)
library(patchwork)

plot_data <-
  read_tsv('data/variant_filter_benchmark_pools.tsv') %>% 
  filter(filtering != 'baseline') %>% 
  filter(variant_type == 'all') %>% 
  mutate(caller = if_else(caller_type == 'joint', 'GATK joint', 'GATK')) %>% 
  mutate(direction = if_else(((metric == 'F1' | metric == 'sensitivity') & (value - baseline > 0)) |
                               (metric == 'FDR' & (value - baseline < 0)), 'up', 'down')) %>% 
  mutate(filtering = factor(filtering, levels = c('HT', 'ML-S', 'ML-F1', 'VQSR')),
         caller_position = interaction(caller, filtering))


sens_plot <- 
  plot_data %>% 
  filter(metric == 'sensitivity') %>% 
  ggplot(aes(x=caller_position, y=value, label=filtering)) +
  geom_label(nudge_x = 0.05, hjust = 0, size = 3) +
  geom_point(aes(x=caller_position,y=baseline, color = "Baseline")) +
  geom_segment(aes(y=baseline, yend=value, color=direction),
               arrow = arrow(length = unit(0.015, "npc")),
               size = 0.5,
               show.legend = FALSE) +
  scale_color_manual(name = NULL,
                     values = c("up" = "darkgreen", "down" = "darkred", "Baseline" = "black"),
                     breaks = c("Baseline")) +
  update_geom_defaults("label", list(size = 3.5)) +
  scale_x_discrete(labels = function(x) sub("\\..*$", "", x)) +
  theme_bw() +
  theme(axis.title = element_text(size = 10),
        axis.text = element_text(size = 10),
        legend.position = '') +
  ylim(0.5,1) +
  labs(x=NULL,
       y=NULL) +
  ggtitle('Sensitivity')

fdr_plot <- 
  plot_data %>% 
  filter(metric == 'FDR') %>% 
  ggplot(aes(x=caller_position, y=value, label=filtering)) +
  geom_label(nudge_x = 0.05, hjust = 0, size = 3) +
  geom_point(aes(x=caller_position,y=baseline, color = "Baseline")) +
  geom_segment(aes(y=baseline, yend=value, color=direction),
               arrow = arrow(length = unit(0.015, "npc")),
               size = 0.5,
               show.legend = FALSE) +
  scale_color_manual(name = NULL,
                     values = c("up" = "darkgreen", "down" = "darkred", "Baseline" = "black"),
                     breaks = c("Baseline")) +
  update_geom_defaults("label", list(size = 4)) +
  scale_x_discrete(labels = function(x) sub("\\..*$", "", x)) +
  theme_bw() +
  theme(axis.title = element_text(size = 10),
        axis.text = element_text(size = 10),
        legend.position = '') +
  ylim(0,0.5) +
  labs(x=NULL,
       y=NULL) +
  ggtitle('FDR')

f1_plot <- 
  plot_data %>% 
  filter(metric == 'F1') %>% 
  ggplot(aes(x=caller_position, y=value, label=filtering)) +
  geom_label(nudge_x = 0.05, hjust = 0, size = 3) +
  geom_point(aes(x=caller_position,y=baseline, color = "Baseline")) +
  geom_segment(aes(y=baseline, yend=value, color=direction),
               arrow = arrow(length = unit(0.015, "npc")),
               size = 0.5,
               show.legend = FALSE) +
  scale_color_manual(name = NULL,
                     values = c("up" = "darkgreen", "down" = "darkred", "Baseline" = "black"),
                     breaks = c("Baseline")) +
  update_geom_defaults("label", list(size = 4)) + 
  scale_x_discrete(labels = function(x) sub("\\..*$", "", x)) +
  theme_bw() +
  theme(axis.title = element_text(size = 10),
        axis.text = element_text(size = 10),
        legend.position = '') +
  ylim(0.5,1) +
  labs(x=NULL,
       y=NULL) +
  ggtitle('F1-score')

bench_plot <- 
  sens_plot | fdr_plot | f1_plot + theme()

ggsave('./variant_filter_benchmark_pool.png',
       bench_plot,
       width = 12,
       height = 4)