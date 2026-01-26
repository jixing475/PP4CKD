# Launch PPB3 Shiny App

cat("\n")
cat("======================================\n")
cat("  PP4CKD Shiny App Launcher\n")
cat("======================================\n\n")

# Set environment
Sys.setenv(RETICULATE_PYTHON = ".venv-shiny/bin/python")

# Check dependencies
cat("Checking dependencies...\n")
required_packages <- c("shiny", "reticulate", "tidyverse", "DT", "plotly", "shinyjs", "writexl")
missing <- c()

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    missing <- c(missing, pkg)
  }
}

if (length(missing) > 0) {
  cat(sprintf("❌ Missing packages: %s\n", paste(missing, collapse = ", ")))
  cat("Please install them first.\n")
  quit(status = 1)
}

cat("✓ All dependencies found\n\n")

# Launch app
cat("Launching Shiny app...\n")
cat("The app will open in your default web browser.\n")
cat("Press Ctrl+C to stop the server.\n\n")

library(shiny)
runApp(".", port = 8080, launch.browser = TRUE)
