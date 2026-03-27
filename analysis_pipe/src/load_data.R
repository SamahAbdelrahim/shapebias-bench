library(readr)
library(dplyr)

# ---------------------------------------------------------------------------
# Paths from .env
# ---------------------------------------------------------------------------
# here::here() points to analysis_pipe/ (renv project root).
# Repo root is one level up.
REPO_ROOT <- normalizePath(file.path(here::here(), ".."))

read_dotenv <- function(path = file.path(REPO_ROOT, ".env")) {
  if (!file.exists(path)) stop(".env file not found at: ", path)
  lines <- readLines(path, warn = FALSE)
  lines <- lines[grepl("=", lines) & !grepl("^\\s*#", lines)]
  for (line in lines) {
    parts <- strsplit(line, "=", fixed = TRUE)[[1]]
    key   <- trimws(parts[1])
    value <- trimws(paste(parts[-1], collapse = "="))
    do.call(Sys.setenv, setNames(list(value), key))
  }
}

#' Get the results directory (absolute) from .env RESULTS_DIR
get_results_dir <- function() {
  read_dotenv()
  rd <- Sys.getenv("RESULTS_DIR", unset = "results")
  # Resolve relative paths against repo root
  if (!startsWith(rd, "/")) rd <- file.path(REPO_ROOT, rd)
  rd
}

#' Get the figures output directory, creating it if needed
get_figures_dir <- function() {
  fig_dir <- file.path(get_results_dir(), "figures")
  if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
  fig_dir
}

#' Get the default CSV data path
get_data_path <- function(filename = "local_eval.csv") {
  file.path(get_results_dir(), "data", filename)
}

# Model sizes in billions of parameters
MODEL_SIZES <- c(
  "qwen3.5-0.8b"     = 0.8,
  "internvl"          = 2,
  "qwen3-vl-2b"      = 2,
  "smolvlm"           = 2.2,
  "tinyllava"         = 3.1,
  "qwen3-vl-4b"      = 4,
  "qwen3.5-4b"        = 4,
  "qwen3.5-9b"        = 9,
  "qwen3.5-27b"       = 27,
  "qwen3.5-35b-a3b"   = 35,
  "llama4-scout"       = 109,
  "qwen3.5-122b-a10b" = 122
)

# Model family mapping (for color-coding plots)
MODEL_FAMILIES <- c(
  "qwen3.5-0.8b"     = "Qwen",
  "internvl"          = "InternVL",
  "qwen3-vl-2b"      = "Qwen",
  "smolvlm"           = "SmolVLM",
  "tinyllava"         = "TinyLLaVA",
  "qwen3-vl-4b"      = "Qwen",
  "qwen3.5-4b"        = "Qwen",
  "qwen3.5-9b"        = "Qwen",
  "qwen3.5-27b"       = "Qwen",
  "qwen3.5-35b-a3b"   = "Qwen",
  "llama4-scout"       = "LLaMA",
  "qwen3.5-122b-a10b" = "Qwen"
)

FAMILY_COLORS <- c(
  "Qwen"     = "#1f77b4",
  "InternVL" = "#ff7f0e",
  "SmolVLM"  = "#2ca02c",
  "TinyLLaVA"= "#9467bd",
  "LLaMA"    = "#d62728"
)

#' Load and clean results CSV(s)
load_results <- function(csv_paths) {
  if (length(csv_paths) == 1) {
    df <- read_csv(csv_paths, show_col_types = FALSE)
  } else {
    df <- bind_rows(lapply(csv_paths, read_csv, show_col_types = FALSE))
  }
  df <- df |>
    mutate(
      choice = factor(choice, levels = c("shape", "texture", "unclear")),
      order_method = ifelse(order_method == "fixed", "deterministic", order_method)
    )
  df
}

# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------
#' Filter trials by ordering, order_method, and/or word_type
filter_trials <- function(df, ordering = NULL, order_method = NULL, word_type = NULL) {
  if (!is.null(ordering))     df <- df |> filter(ordering %in% !!ordering)
  if (!is.null(order_method)) df <- df |> filter(order_method %in% !!order_method)
  if (!is.null(word_type))    df <- df |> filter(word_type %in% !!word_type)
  df
}

#' Build a subtitle string describing active filters (NULL if no filters)
make_filter_subtitle <- function(ordering = NULL, order_method = NULL, word_type = NULL) {
  parts <- c()
  if (!is.null(ordering))     parts <- c(parts, paste("Ordering:", paste(ordering, collapse = ", ")))
  if (!is.null(order_method)) parts <- c(parts, paste("Method:", paste(order_method, collapse = ", ")))
  if (!is.null(word_type))    parts <- c(parts, paste("Word type:", paste(word_type, collapse = ", ")))
  if (length(parts) == 0) return(NULL)
  paste(parts, collapse = " | ")
}

#' Add model size info and create ordered factor for plotting
add_model_size <- function(df) {
  size_df <- tibble(
    model = names(MODEL_SIZES),
    param_b = unname(MODEL_SIZES)
  )
  df <- df |>
    left_join(size_df, by = "model") |>
    mutate(
      model_label = paste0(model, "\n(", param_b, "B)"),
      model_label = factor(
        model_label,
        levels = size_df |>
          arrange(param_b) |>
          mutate(label = paste0(model, "\n(", param_b, "B)")) |>
          pull(label)
      ),
      model_family = MODEL_FAMILIES[model]
    )
  df
}
