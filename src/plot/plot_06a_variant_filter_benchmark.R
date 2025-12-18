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

titlesize <- 8
textsize <- 7
subtitlesize <- 8

plot_data <-
  read_tsv('data/variant_filter_benchmark_pools.tsv', show_col_types = F) %>% 
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
  ggplot(aes(x=caller_position, y=value)) +
  geom_point(aes(x=caller_position,y=baseline, color = "Baseline"), size = 0.7) +
  geom_segment(aes(y=baseline, yend=value, color=direction),
               arrow = arrow(length = unit(0.02, "npc")),
               size = 0.7,
               show.legend = FALSE) +
  scale_color_manual(name = NULL,
                     values = c("up" = "darkgreen", "down" = "darkred", "Baseline" = "black"),
                     breaks = c("Baseline")) +
  scale_x_discrete(labels = function(x) sub("^.*\\.", "", x)) +
  theme_bw() +
  theme(axis.title = element_text(size = textsize),
        axis.text = element_text(size = textsize),
        plot.subtitle=element_text(size = subtitlesize),
        plot.title=element_text(size = titlesize),
        legend.position = '') +
  ylim(0.5,1) +
  labs(x=NULL,
       y=NULL) +
  ggtitle('Sensitivity')

fdr_plot <- 
  plot_data %>% 
  filter(metric == 'FDR') %>% 
  ggplot(aes(x=caller_position, y=value)) +
  geom_point(aes(x=caller_position,y=baseline, color = "Baseline"), size = 0.7) +
  geom_segment(aes(y=baseline, yend=value, color=direction),
               arrow = arrow(length = unit(0.02, "npc")),
               size = 0.7,
               show.legend = FALSE) +
  scale_color_manual(name = NULL,
                     values = c("up" = "darkgreen", "down" = "darkred", "Baseline" = "black"),
                     breaks = c("Baseline")) +
  scale_x_discrete(labels = function(x) sub("^.*\\.", "", x)) +
  theme_bw() +
  theme(axis.title = element_text(size = textsize),
        axis.text = element_text(size = textsize),
        plot.subtitle=element_text(size = subtitlesize),
        plot.title=element_text(size = titlesize),
        legend.position = '') +
  ylim(0,0.5) +
  labs(x=NULL,
       y=NULL) +
  ggtitle('FDR')

f1_plot <- 
  plot_data %>% 
  filter(metric == 'F1') %>% 
  ggplot(aes(x=caller_position, y=value)) +
  geom_point(aes(x=caller_position,y=baseline, color = "Baseline"), size = 0.7) +
  geom_segment(aes(y=baseline, yend=value, color=direction),
               arrow = arrow(length = unit(0.02, "npc")),
               size = 0.7,
               show.legend = FALSE) +
  scale_color_manual(name = NULL,
                     values = c("up" = "darkgreen", "down" = "darkred", "Baseline" = "black"),
                     breaks = c("Baseline")) +
  scale_x_discrete(labels = function(x) sub("^.*\\.", "", x)) +
  theme_bw() +
  theme(axis.title = element_text(size = textsize),
        axis.text = element_text(size = textsize),
        plot.subtitle=element_text(size = subtitlesize),
        plot.title=element_text(size = titlesize),
        legend.position = '') +
  ylim(0.5,1) +
  labs(x=NULL,
       y=NULL) +
  ggtitle('F1-score')

bench_plot <- 
  sens_plot | fdr_plot | f1_plot + theme()

ggsave('./variant_filter_benchmark_pool.png',
       bench_plot,
       dpi=600,
       width = 178,
       height = 118.66/2,
       units = "mm")