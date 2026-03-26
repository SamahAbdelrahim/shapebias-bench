library(ggplot2)
library(dplyr)
library(scales)

SHAPE_COL   <- "#6B8E23"
TEXTURE_COL <- "#FFB6C1"
UNCLEAR_COL <- "#AAAAAA"

#' Main plot: proportion shape choice by model
plot_model_bias_main <- function(df, ordering = NULL, order_method = NULL, word_type = NULL) {
  df <- filter_trials(df, ordering = ordering, order_method = order_method, word_type = word_type)
  subtitle <- make_filter_subtitle(ordering = ordering, order_method = order_method, word_type = word_type)

  summary_df <- df |>
    group_by(model_label) |>
    summarise(
      n = n(),
      shape_prop = mean(choice == "shape", na.rm = TRUE),
      .groups = "drop"
    )

  p <- ggplot(summary_df, aes(x = model_label, y = shape_prop)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    geom_point(colour = SHAPE_COL, size = 4) +
    geom_text(aes(label = sprintf("%.2f", shape_prop)),
              vjust = -1.2, size = 3.5) +
    scale_y_continuous(limits = c(0, 1), labels = label_percent()) +
    labs(
      x = "Model (parameters)",
      y = "Proportion shape choice",
      title = "Shape bias by model",
      subtitle = subtitle
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 10))

  p
}

#' Supplementary plot: stacked breakdown by model
plot_model_bias_supplement <- function(df, ordering = NULL, order_method = NULL, word_type = NULL) {
  df <- filter_trials(df, ordering = ordering, order_method = order_method, word_type = word_type)
  subtitle <- make_filter_subtitle(ordering = ordering, order_method = order_method, word_type = word_type)

  summary_df <- df |>
    group_by(model_label, choice) |>
    summarise(n = n(), .groups = "drop_last") |>
    mutate(prop = n / sum(n)) |>
    ungroup()

  p <- ggplot(summary_df, aes(x = model_label, y = prop, fill = choice)) +
    geom_col(width = 0.7) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    scale_fill_manual(
      values = c(shape = SHAPE_COL, texture = TEXTURE_COL, unclear = UNCLEAR_COL),
      name = "Choice"
    ) +
    scale_y_continuous(labels = label_percent()) +
    labs(
      x = "Model (parameters)",
      y = "Proportion",
      title = "Response breakdown by model",
      subtitle = subtitle
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 10))

  p
}
