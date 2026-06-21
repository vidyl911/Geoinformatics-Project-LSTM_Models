# =============================================================================
# SCRIPT 02: DATA PREPROCESSING & FEATURE ENGINEERING
# Methodology: Section 3.2.1
# =============================================================================
# Run after: 01_setup_and_data_loading.R
# Outputs: df_clean.rds, df_scaled.rds, df_supervised.rds

source("F:\\Geoinformatics Project\\01_setup_and_data_loading.R")

df <- readRDS(file.path(OUTPUT_PATH, "df_raw.rds"))

# ── 1. Missing Value Audit ─────────────────────────────────────────────────────
cat("\n══ MISSING VALUE AUDIT ═══════════════════════════════════\n")

miss_summary <- df %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") %>%
  mutate(
    pct_missing = round(n_missing / nrow(df) * 100, 2),
    flag = case_when(
      pct_missing == 0        ~ "✅ Complete",
      pct_missing <= 5        ~ "⚠ Minor (<5%)",
      pct_missing <= 20       ~ "⚡ Moderate (<20%)",
      TRUE                    ~ "🔴 Severe (>20%)"
    )
  ) %>%
  arrange(desc(n_missing))

print(miss_summary, n = 30)

# ── 2. Temporal Interpolation for NDVI, NDMI ─────────────────────────────────
# Methodology §3.2.1: "Missing values addressed using temporal interpolation"
cat("\n── Temporal interpolation (by spatial point if ID available)…\n")

# FIX: detect 'pid' as the point ID column (Italian dataset uses 'pid')
col_id <- detect_col(df, c("pid", "point", "site", "id", "pixel", "fid", "objectid", "loc"))

interpolate_ts <- function(x) {
  # Linear interpolation via zoo; fall-back to spline
  out <- zoo::na.approx(x, na.rm = FALSE)
  # Fill any remaining NA at edges with spline
  if (any(is.na(out))) out <- zoo::na.spline(out, na.rm = FALSE)
  # Final fill: forward/back carry
  out <- zoo::na.locf(out, na.rm = FALSE, fromLast = FALSE)
  out <- zoo::na.locf(out, na.rm = FALSE, fromLast = TRUE)
  out
}

target_cols <- c("ndvi", "ndmi", "rainfall", "temperature", "humidity")
target_cols <- target_cols[target_cols %in% names(df)]

if (!is.null(col_id) && col_id %in% names(df)) {
  df <- df %>%
    arrange(.data[[col_id]], date) %>%
    group_by(.data[[col_id]]) %>%
    mutate(across(all_of(target_cols), interpolate_ts)) %>%
    ungroup()
} else {
  df <- df %>%
    arrange(date) %>%
    mutate(across(all_of(target_cols), interpolate_ts))
}

# Post-interpolation audit
miss_after <- df %>%
  select(all_of(target_cols)) %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "remaining_NA")

cat("\n  Missing values after interpolation:\n")
print(miss_after)

# ── 3. Remove Physically Impossible Values ────────────────────────────────────
df <- df %>%
  mutate(
    ndvi = if_else(ndvi < -1 | ndvi > 1, NA_real_, ndvi),
    ndmi = if_else(ndmi < -1 | ndmi > 1, NA_real_, ndmi)
  )
if ("rainfall"    %in% names(df)) df <- df %>% mutate(rainfall    = if_else(rainfall < 0, NA_real_, rainfall))
if ("humidity"    %in% names(df)) df <- df %>% mutate(humidity    = if_else(humidity < 0 | humidity > 100, NA_real_, humidity))

# Second-pass interpolation after physical outlier removal
df <- df %>% mutate(across(all_of(target_cols), interpolate_ts))

df_clean <- df %>% filter(if_any(all_of(target_cols), ~ !is.na(.)))
cat("\n✅ Clean rows: ", format(nrow(df_clean), big.mark = ","), "\n")

# ── 4. Min-Max Normalisation ──────────────────────────────────────────────────
# Methodology §3.2.1: "normalised using Min-Max scaling"
cat("\n── Min-Max Normalisation…\n")

scaler_stats <- df_clean %>%
  select(all_of(target_cols)) %>%
  summarise(across(everything(), list(min = min, max = max), na.rm = TRUE)) %>%
  pivot_longer(everything(),
               names_to  = c("variable", ".value"),
               names_sep = "_(?=[^_]+$)")

# Save scaler for inverse-transform later
saveRDS(scaler_stats, file.path(OUTPUT_PATH, "scaler_stats.rds"))

min_max_scale <- function(x, mn, mx) {
  if (mx == mn) return(rep(0, length(x)))
  (x - mn) / (mx - mn)
}

df_scaled <- df_clean
for (v in target_cols) {
  mn <- scaler_stats$min[scaler_stats$variable == v]
  mx <- scaler_stats$max[scaler_stats$variable == v]
  df_scaled[[paste0(v, "_sc")]] <- min_max_scale(df_scaled[[v]], mn, mx)
}

scaled_cols <- paste0(target_cols, "_sc")
cat("  Scaled columns created:", paste(scaled_cols, collapse = ", "), "\n")

# ── 5. Lagged Feature Engineering ────────────────────────────────────────────
# Methodology §3.2.1: "Lagged variables including previous values of NDVI, NDMI"
cat("\n── Creating lagged features (lags 1, 3, 7, 14, 30 days)…\n")

lag_vars  <- c("ndvi_sc", "ndmi_sc", "rainfall_sc", "temperature_sc", "humidity_sc")
lag_vars  <- lag_vars[lag_vars %in% names(df_scaled)]
lag_steps <- c(1, 3, 7, 14, 30)

add_lags <- function(data, vars, lags, group_col = NULL) {
  for (v in vars) {
    for (l in lags) {
      col_name <- paste0(v, "_lag", l)
      if (!is.null(group_col) && group_col %in% names(data)) {
        data <- data %>%
          group_by(.data[[group_col]]) %>%
          mutate(!!col_name := lag(.data[[v]], l)) %>%
          ungroup()
      } else {
        data <- data %>% mutate(!!col_name := lag(.data[[v]], l))
      }
    }
  }
  data
}

df_lagged <- add_lags(df_scaled, lag_vars, lag_steps, group_col = col_id)

# Rolling means (7-day and 30-day)
add_rolling <- function(data, vars, windows = c(7, 30), group_col = NULL) {
  for (v in vars) {
    for (w in windows) {
      col_name <- paste0(v, "_roll", w)
      if (!is.null(group_col) && group_col %in% names(data)) {
        data <- data %>%
          group_by(.data[[group_col]]) %>%
          mutate(!!col_name := zoo::rollmeanr(.data[[v]], k = w, fill = NA)) %>%
          ungroup()
      } else {
        data <- data %>%
          mutate(!!col_name := zoo::rollmeanr(.data[[v]], k = w, fill = NA))
      }
    }
  }
  data
}

df_featured <- add_rolling(df_lagged, lag_vars, windows = c(7, 30), group_col = col_id)

# FIX: use lubridate:: prefix to avoid masking by Hmisc/psych
df_featured <- df_featured %>%
  mutate(
    doy_sin   = sin(2 * pi * doy / 365),
    doy_cos   = cos(2 * pi * doy / 365),
    month_sin = sin(2 * pi * lubridate::month(date) / 12),
    month_cos = cos(2 * pi * lubridate::month(date) / 12)
  )

# Topographic: mountain flag numeric
if ("terrain" %in% names(df_featured)) {
  df_featured <- df_featured %>%
    mutate(is_mountain = as.integer(terrain == "Mountain"))
}

cat("  Feature matrix: ", ncol(df_featured), " columns\n")
cat("  Feature engineering complete.\n")

# ── 6. Supervised Learning Format ────────────────────────────────────────────
# Drop rows with any NA in predictors (from lags)
predictor_pattern <- c("_sc$", "_lag", "_roll", "_sin$", "_cos$",
                       "is_mountain", "elevation")
predictor_cols <- grep(paste(predictor_pattern, collapse = "|"),
                       names(df_featured), value = TRUE)
predictor_cols <- predictor_cols[!grepl("^ndvi_|^ndmi_", predictor_cols)]

df_supervised <- df_featured %>%
  select(date, year, season, month, doy,
         all_of(c("fuel_class", "terrain")[c("fuel_class","terrain") %in% names(.)]),
         any_of(c("elevation", "is_mountain", "pid", "slope",
                  "northerness", "northerness_slope")),  # keep useful spatial cols
         all_of(c("ndvi", "ndmi")),          # raw targets
         all_of(c("ndvi_sc", "ndmi_sc")),    # scaled targets
         all_of(predictor_cols)) %>%
  filter(complete.cases(select(., all_of(c("ndvi_sc","ndmi_sc")))))

cat("\n✅ Supervised dataset: ", format(nrow(df_supervised), big.mark=","),
    " rows × ", ncol(df_supervised), " columns\n")

# ── 7. Save ───────────────────────────────────────────────────────────────────
saveRDS(df_clean,      file.path(OUTPUT_PATH, "df_clean.rds"))
saveRDS(df_scaled,     file.path(OUTPUT_PATH, "df_scaled.rds"))
saveRDS(df_supervised, file.path(OUTPUT_PATH, "df_supervised.rds"))

cat("\n💾 Saved: df_clean.rds | df_scaled.rds | df_supervised.rds\n")
cat("Script 02 complete ✓\n")

