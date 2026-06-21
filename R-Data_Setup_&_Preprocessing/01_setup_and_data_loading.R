# =============================================================================
# SCRIPT 01: SETUP & DATA LOADING
# Study: NDVI/NDMI Prediction using LSTM — Lombardy Region
# Methodology: Section 3.1 — Study Area and Dataset Description
# =============================================================================

# ── 1. Package Management ─────────────────────────────────────────────────────
required_packages <- c(
  # Data wrangling
  "tidyverse", "data.table", "lubridate", "janitor",
  # Geospatial
  "sf", "terra", "tmap", "leaflet", "leaflet.extras",
  "mapview", "ggmap", "ggspatial", "RColorBrewer",
  # Statistics & EDA
  "moments", "corrplot", "ggcorrplot", "Hmisc", "psych",
  "scales", "patchwork", "ggridges", "ggbeeswarm",
  # Time series
  "forecast", "tseries", "zoo", "xts",
  # ML / Modelling
  "randomForest", "e1071", "caret", "Metrics",
  # Visualization extras
  "viridis", "ggnewscale", "ggtext", "showtext",
  "gridExtra", "cowplot", "ggpubr", "plotly",
  "htmlwidgets", "DT", "kableExtra",
  # Reporting
  "gt", "gtExtras"
)

# Install missing packages silently
install_if_missing <- function(pkgs) {
  missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(missing) > 0) {
    message("Installing: ", paste(missing, collapse = ", "))
    install.packages(missing, dependencies = TRUE, quiet = TRUE)
  }
}
install_if_missing(required_packages)

# Load all packages
suppressPackageStartupMessages(
  invisible(lapply(required_packages, library, character.only = TRUE))
)

# ── 2. Google Fonts ───────────────────────────────────────────────────────────
font_add_google("Montserrat", "montserrat")
font_add_google("Fira Code",  "firacode")
font_add_google("Playfair Display", "playfair")
showtext_auto()

# ── 3. Global Theme ───────────────────────────────────────────────────────────
# FIX 1: Use ggplot2::margin() explicitly to avoid masking by Hmisc/psych
theme_lombardy <- function(base_size = 12) {
  theme_minimal(base_size = base_size, base_family = "montserrat") +
    theme(
      plot.background    = element_rect(fill = "#0d1117", color = NA),
      panel.background   = element_rect(fill = "#161b22", color = NA),
      panel.grid.major   = element_line(color = "#21262d", linewidth = 0.4),
      panel.grid.minor   = element_blank(),
      axis.text          = element_text(color = "#8b949e", size = rel(0.85)),
      axis.title         = element_text(color = "#c9d1d9", face = "bold"),
      plot.title         = element_text(color = "#e6edf3", size = rel(1.4),
                                        face = "bold", family = "playfair",
                                        margin = ggplot2::margin(b = 6)),
      plot.subtitle      = element_text(color = "#8b949e", size = rel(0.95),
                                        margin = ggplot2::margin(b = 12)),
      plot.caption       = element_text(color = "#484f58", size = rel(0.75),
                                        hjust = 1),
      legend.background  = element_rect(fill = "#161b22", color = NA),
      legend.text        = element_text(color = "#8b949e"),
      legend.title       = element_text(color = "#c9d1d9", face = "bold"),
      strip.background   = element_rect(fill = "#21262d", color = NA),
      strip.text         = element_text(color = "#c9d1d9", face = "bold"),
      plot.margin        = ggplot2::margin(16, 16, 16, 16)
    )
}
theme_set(theme_lombardy())

# Colour palettes
pal_ndvi  <- viridis::viridis(10, option = "D")
pal_ndmi  <- viridis::viridis(10, option = "E")
pal_div   <- RColorBrewer::brewer.pal(11, "RdYlGn")
pal_fuel  <- c("#58a6ff","#3fb950","#f78166","#d2a8ff",
               "#ff7b72","#ffa657","#79c0ff","#56d364")

# ── 4. Directory Structure ────────────────────────────────────────────────────
DATA_PATH   <- "F:/Geoinformatics Project/csv_gapfilled (2)"
OUTPUT_PATH <- "F:/Geoinformatics Project/outputs"
PLOT_PATH   <- file.path(OUTPUT_PATH, "plots")
MODEL_PATH  <- file.path(OUTPUT_PATH, "models")
TABLE_PATH  <- file.path(OUTPUT_PATH, "tables")

for (d in c(OUTPUT_PATH, PLOT_PATH, MODEL_PATH, TABLE_PATH)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# ── 5. Data Loading ───────────────────────────────────────────────────────────
message("\n📂 Scanning data directory: ", DATA_PATH)

csv_files <- list.files(DATA_PATH, pattern = "\\.csv$",
                        full.names = TRUE, recursive = TRUE)
message("   Found ", length(csv_files), " CSV files")

# Read & bind all CSVs with progress
read_with_progress <- function(files) {
  pb <- txtProgressBar(min = 0, max = length(files), style = 3)
  dfs <- vector("list", length(files))
  for (i in seq_along(files)) {
    dfs[[i]] <- tryCatch(
      fread(files[i], na.strings = c("", "NA", "NaN", "null", "-9999")),
      error = function(e) { message("⚠ Could not read: ", basename(files[i])); NULL }
    )
    setTxtProgressBar(pb, i)
  }
  close(pb)
  rbindlist(Filter(Negate(is.null), dfs), fill = TRUE)
}

raw_dt <- read_with_progress(csv_files)
message("\n✅ Raw dataset loaded: ", 
        format(nrow(raw_dt), big.mark = ","), " rows × ", 
        ncol(raw_dt), " columns")

# ── 6. Column Standardisation ─────────────────────────────────────────────────
# Clean names (lowercase, no spaces)
setnames(raw_dt, janitor::make_clean_names(names(raw_dt)))

# Auto-detect key columns (flexible to various naming conventions)
detect_col <- function(dt, patterns) {
  found <- NULL
  for (p in patterns) {
    m <- grep(p, names(dt), ignore.case = TRUE, value = TRUE)
    if (length(m)) { found <- m[1]; break }
  }
  found
}

col_date  <- detect_col(raw_dt, c("date","time","day","timestamp"))
col_ndvi  <- detect_col(raw_dt, c("ndvi"))
col_ndmi  <- detect_col(raw_dt, c("ndmi"))
col_rain  <- detect_col(raw_dt, c("rain","precip","rainfall","prec",
                                  "precipitazione"))                  # Italian
col_temp  <- detect_col(raw_dt, c("temp","tmax","tmin","tmean","temperature",
                                  "temperatura"))                     # Italian
col_hum   <- detect_col(raw_dt, c("hum","humidity","rh","relative_humidity",
                                  "umidita","umidita_relativa"))      # Italian
col_fuel  <- detect_col(raw_dt, c("fuel","fuel_class","fuelclass","fire_fuel",
                                  "fire_fuel_class"))                 # Italian
col_lon   <- detect_col(raw_dt, c("lon","longitude","lng"))            # x excluded (ambiguous)
col_lat   <- detect_col(raw_dt, c("lat","latitude"))                   # y excluded (ambiguous)
col_elev  <- detect_col(raw_dt, c("elev","elevation","dem","altitude","alt",
                                  "elevazione"))                      # Italian

cat("\n── Column Mapping ──────────────────────────────────────\n")

# FIX 2: Use NA as fallback for undetected columns so lengths always match
mapping <- data.frame(
  Variable = c("Date","NDVI","NDMI","Rainfall","Temperature",
               "Humidity","Fuel Class","Longitude","Latitude","Elevation"),
  Detected = c(
    col_date  %||% NA_character_,
    col_ndvi  %||% NA_character_,
    col_ndmi  %||% NA_character_,
    col_rain  %||% NA_character_,
    col_temp  %||% NA_character_,
    col_hum   %||% NA_character_,
    col_fuel  %||% NA_character_,
    col_lon   %||% NA_character_,
    col_lat   %||% NA_character_,
    col_elev  %||% NA_character_
  ),
  stringsAsFactors = FALSE
)
print(mapping)

# Note: lon/lat are absent from this dataset (point ID 'pid' is used instead)
undetected <- mapping$Variable[is.na(mapping$Detected)]
expected_absent <- c("Longitude", "Latitude")   # confirmed not in this dataset
unexpected_absent <- setdiff(undetected, expected_absent)
if (length(unexpected_absent) > 0)
  message("⚠ Could not detect columns for: ", paste(unexpected_absent, collapse = ", "))
if (length(intersect(undetected, expected_absent)) > 0)
  message("ℹ  No lon/lat columns found — spatial joins will use 'pid' as pixel ID")

# ── 7. Type Casting & Date Parsing ────────────────────────────────────────────
df <- as_tibble(raw_dt)

if (!is.null(col_date)) {
  df <- df %>%
    mutate(
      date   = parse_date_time(.data[[col_date]],
                               orders = c("ymd","dmy","mdy","ymd HMS"),
                               quiet  = TRUE),
      year   = lubridate::year(date),
      # FIX 3: Use lubridate:: prefix to avoid masking by Hmisc/psych
      month  = lubridate::month(date, label = TRUE, abbr = TRUE),
      doy    = lubridate::yday(date),
      season = case_when(
        lubridate::month(date) %in% c(12, 1, 2) ~ "Winter",
        lubridate::month(date) %in% c(3, 4, 5)  ~ "Spring",
        lubridate::month(date) %in% c(6, 7, 8)  ~ "Summer",
        TRUE                                     ~ "Autumn"
      ) %>% factor(levels = c("Spring","Summer","Autumn","Winter"))
    )
}

# Ensure numeric targets
for (col in c(col_ndvi, col_ndmi, col_rain, col_temp, col_hum, col_elev)) {
  if (!is.null(col) && col %in% names(df)) {
    df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
  }
}

# Rename to standard internal names for pipeline
rename_map <- c(
  ndvi = col_ndvi, ndmi = col_ndmi, rainfall = col_rain,
  temperature = col_temp, humidity = col_hum, fuel_class = col_fuel,
  lon = col_lon, lat = col_lat, elevation = col_elev
)
rename_map <- rename_map[!sapply(rename_map, is.null)]
df <- df %>% rename(any_of(rename_map))

# Mountain classification (if elevation available)
if ("elevation" %in% names(df)) {
  df <- df %>%
    mutate(terrain = if_else(elevation >= 600, "Mountain", "Lowland") %>%
             factor())
}

message("\n✅ Dataset structured: ", 
        format(nrow(df), big.mark = ","), " rows")
message("   Date range : ", min(df$date, na.rm = TRUE), 
        " → ", max(df$date, na.rm = TRUE))
message("   Fuel classes: ", 
        if ("fuel_class" %in% names(df)) 
          length(unique(df$fuel_class)) else "not found")

# ── 8. Save Processed Object ──────────────────────────────────────────────────
saveRDS(df,         file.path(OUTPUT_PATH, "df_raw.rds"))
saveRDS(col_ndvi,   file.path(OUTPUT_PATH, "col_ndvi.rds"))
saveRDS(col_ndmi,   file.path(OUTPUT_PATH, "col_ndmi.rds"))

message("\n💾 Saved: df_raw.rds")
message("Script 01 complete ✓\n")

source("01_setup_and_data_loading.R")
getwd()
list.files()
