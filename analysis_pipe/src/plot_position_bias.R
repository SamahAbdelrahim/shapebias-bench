library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

TRACKS_COL  <- "#4CAF50"
BIAS_COL    <- "#F44336"
MISSING_COL <- "#AAAAAA"

#' Classify each stimulus x word x model pair as "tracks image" or "position bias"
#'
#' Requires both orderings (shape_first + texture_first) to be present.
#' Logic: if the model picks image-1 when shape is first and image-2 when shape
#' is second (or vice-versa for texture), it is tracking the image content.
#' If it gives the same answer regardless of ordering, it has position bias.
classify_position_bias <- function(df) {
  paired <- df |>
    filter(ordering %in% c("shape_first", "texture_first"),
           order_method == "deterministic") |>
    select(model, model_label, stim_id, word, ordering, parsed_answer) |>
    pivot_wider(names_from = ordering, values_from = parsed_answer,
                values_fn = list)

  # Keep only rows where we have exactly one answer per ordering
  paired <- paired |>
    filter(
      lengths(shape_first) == 1,
      lengths(texture_first) == 1
    ) |>
    mutate(
      shape_first = unlist(shape_first),
      texture_first = unlist(texture_first)
    )

  paired |>
    mutate(
      validity = case_when(
        # shape_first: 1=shape,2=texture; texture_first: 1=texture,2=shape
        # Tracks shape: picks shape position in both orderings
        shape_first == "1" & texture_first == "2" ~ "tracks_image",
        # Tracks texture: picks texture position in both orderings
        shape_first == "2" & texture_first == "1" ~ "tracks_image",
        # Same answer regardless → position bias
        shape_first == texture_first ~ "position_bias",
        TRUE ~ "other"
      )
    )
}

#' Main plot: proportion of trials that track image vs show position bias, by model
plot_position_bias_main <- function(df) {
  classified <- classify_position_bias(df)

  summary_df <- classified |>
    group_by(model_label, validity) |>
    summarise(n = n(), .groups = "drop_last") |>
    mutate(prop = n / sum(n), total = sum(n)) |>
    ungroup()

  # Add label for tracks_image proportion
  label_df <- summary_df |>
    filter(validity == "tracks_image") |>
    select(model_label, prop)

  ggplot(summary_df, aes(x = model_label, y = prop, fill = validity)) +
    geom_col(width = 0.7) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    geom_text(data = label_df,
              aes(x = model_label, y = prop, label = sprintf("%.0f%%", prop * 100),
                  fill = NULL),
              vjust = -0.5, size = 3.5) +
    scale_fill_manual(
      values = c(tracks_image = TRACKS_COL, position_bias = BIAS_COL, other = MISSING_COL),
      labels = c(tracks_image = "Tracks image", position_bias = "Position bias", other = "Other"),
      name = "Validity"
    ) +
    scale_y_continuous(limits = c(0, 1.08), labels = label_percent()) +
    labs(
      x = "Model (parameters)",
      y = "Proportion of paired trials",
      title = "Position bias validation",
      subtitle = "Do models track the correct image when positions are swapped?"
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 10))
}

#' Supplementary: position bias broken down by stimulus
plot_position_bias_by_stimulus <- function(df) {
  classified <- classify_position_bias(df)

  summary_df <- classified |>
    group_by(model_label, stim_id, validity) |>
    summarise(n = n(), .groups = "drop_last") |>
    mutate(prop = n / sum(n)) |>
    ungroup() |>
    filter(validity == "tracks_image") |>
    mutate(stim_id = factor(stim_id))

  ggplot(summary_df, aes(x = stim_id, y = prop, colour = model_label)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    geom_point(size = 2.5, position = position_dodge(width = 0.5)) +
    scale_y_continuous(limits = c(0, 1), labels = label_percent()) +
    scale_colour_brewer(palette = "Set2", name = "Model") +
    labs(
      x = "Stimulus ID",
      y = "Proportion tracking image",
      title = "Image tracking by stimulus and model"
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 9))
}
