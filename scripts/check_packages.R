# Check and install required R packages

required_packages <- c("shiny", "reticulate", "tidyverse", "DT", "plotly", "shinyjs", "writexl")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  } else {
    cat(sprintf("✓ %s is already installed\n", pkg))
  }
}

cat("\n✅ All required packages are available!\n")
