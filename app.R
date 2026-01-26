# ==============================================================================
# Drug Repurposing For CKD - Shiny App
# Flask to Shiny Migration
# ==============================================================================

# Configure Python Environment FIRST (before loading reticulate) ----
Sys.setenv(RETICULATE_PYTHON = ".venv-shiny/bin/python")

# Load Required Libraries ----
library(shiny)
library(reticulate)
library(tidyverse)
library(DT)
library(plotly)
library(shinyjs)
library(writexl)

# Verify Python Configuration ----
use_python(".venv-shiny/bin/python", required = TRUE)
cat("Using Python:", py_config()$python, "\n")

# Source Python modules
source_python("scripts/03-fingerprints.py")
source_python("scripts/05-preprocessing.py")
source_python("scripts/06-utils.py")

# Load Target Data and Models ----
cat("Loading target data and models...\n")

# Load target information
target_details <- read.delim("data/external/TARGETSDETAILS_2nd.txt", stringsAsFactors = FALSE)
target_classification <- read.delim("data/external/TARGETCLASSIFICATION_2nd.txt", stringsAsFactors = FALSE)

TARGET_ID_TO_NAME <- setNames(target_details$PREF_NAME, target_details$CHEMBL_ID)
TARGET_ID_TO_CLASS <- setNames(target_classification$CLASS, target_classification$CHEMBL_ID)
TARGET_ID_TO_ORGANISM <- setNames(target_classification$ORGANISM, target_classification$CHEMBL_ID)
TARGET_ID_TO_TYPE <- setNames(target_classification$TYPE, target_classification$CHEMBL_ID)

# Load target labels
target_labels_raw <- readLines("data/external/DNNTARLABELS_2nd.txt")
TARGET_LABELS <- target_labels_raw[-1]  # Skip header

cat(sprintf("✓ Loaded %d targets\n", length(TARGET_ID_TO_NAME)))

# Load models using Python joblib
py_run_string("import joblib")
py_run_string("import numpy as np")

MODELS <- list()
model_types <- c('ECFP4', 'ECFP6', 'AtomPair', 'Layered', 'RDKit', 'MHFP6', 'Fused')
available_models <- c()

for (model_type in model_types) {
  model_file <- sprintf("model/%s_dnn_model_full_data.joblib", tolower(model_type))
  if (file.exists(model_file)) {
    tryCatch({
      py_run_string(sprintf("MODELS_%s = joblib.load('%s')", model_type, model_file))
      MODELS[[model_type]] <- TRUE
      available_models <- c(available_models, model_type)
      file_size <- file.info(model_file)$size / (1024 * 1024)
      cat(sprintf("✓ Loaded %s model (%.2f MB)\n", model_type, file_size))
    }, error = function(e) {
      cat(sprintf("✗ Error loading %s: %s\n", model_type, e$message))
    })
  } else {
    cat(sprintf("⚠ %s model file not found\n", model_type))
  }
}

if (length(MODELS) == 0) {
  stop("❌ No models could be loaded! Please check the model directory.")
}

cat(sprintf("✅ %d models loaded successfully: %s\n",
            length(MODELS), paste(available_models, collapse = ", ")))

# Example Molecules ----
EXAMPLE_MOLECULES <- list(
  Caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
  Ibuprofen = "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
  Benzene = "c1ccccc1",
  Ethanol = "CCO",
  "Acetic_acid" = "CC(=O)O"
)

# UI Definition ----
ui <- fluidPage(
  useShinyjs(),
  includeCSS("www/styles.css"),

  # Include Popper.js (required for Tippy.js)
  tags$head(
    tags$script(src = "https://unpkg.com/@popperjs/core@2"),
    tags$script(src = "https://unpkg.com/tippy.js@6"),
    tags$script(src = "tippy-config.js")
  ),

  # Header
  div(class = "header",
    h1("Drug Repurposing For CKD")
  ),

  # Main Content
  div(class = "main-content",

    # Horizontal Layout: Query Molecules (2/3) + Model Config (1/3) ----
    div(class = "unified-section",
      fluidRow(
        # Left Column: SMILES Input (2/3 width)
        column(8,
          h3("Enter your query molecules:"),
          div(class = "input-section",

          tabsetPanel(id = "input_tabs",

            # Tab 1: Draw Molecule (JSME - LOCAL VERSION)
            tabPanel("Draw Molecule",
              tags$iframe(
                src = "jsme_local.html",
                width = "100%",
                height = "360px",
                frameborder = "0",
                id = "jsme_iframe",
                style = "border: 2px solid #08517f; border-radius: 5px; display: block;"
              ),
              # Completely hidden input for Shiny to receive SMILES from JSME
              shinyjs::hidden(textInput("drawing_smiles", NULL, value = ""))
            ),

            # Tab 2: Manual Input
            tabPanel("Manual Input",
              br(),
              textAreaInput("manual_smiles",
                           "Enter SMILES (one per line):",
                           rows = 10,
                           width = "100%",
                           placeholder = "CCO\nCC(=O)O\nc1ccccc1")
            ),

            # Tab 3: CSV Upload
            tabPanel("Upload CSV",
              br(),
              fileInput("csv_file",
                       "Choose CSV File (must have 'SMILES' column):",
                       accept = ".csv"),
              helpText("Example format: see data/example_input.csv")
            ),

            # Tab 4: Example Molecules
            tabPanel("Example Molecules",
              br(),
              div(class = "example-buttons",
                actionButton("example_caffeine", "Caffeine", class = "btn-example"),
                actionButton("example_ibuprofen", "Ibuprofen", class = "btn-example"),
                actionButton("example_benzene", "Benzene", class = "btn-example"),
                actionButton("example_ethanol", "Ethanol", class = "btn-example"),
                actionButton("example_acetic_acid", "Acetic Acid", class = "btn-example"),
                br(), br(),
                actionButton("load_all_examples", "Load All from example_input.csv",
                            class = "btn-primary")
              ),
              br(),
              textOutput("example_loaded")
            )
          ),

          br(),
          div(style = "text-align: left; display: flex; align-items: center; gap: 10px;",
            actionButton("validate_btn", "Validate SMILES", class = "btn-success btn-large"),
            conditionalPanel(
              condition = "input.input_tabs == 'Draw Molecule'",
              style = "display: inline-block;",
              tags$button("Clear & Redraw",
                         onclick = "document.getElementById('jsme_iframe').contentWindow.clearEditor()",
                         class = "btn btn-large",
                         type = "button",
                         style = "background-color: #dc3545; color: white; font-size: 1.1rem; padding: 10px 30px; border: none; border-radius: 5px;")
            )
          )
        )
      ),

      # Right Column: Model Configuration (1/3 width)
      column(4,
        h3("Model Configuration"),
        div(id = "model_config_section",
          div(class = "model-config",
            selectInput("model_type",
                       "Select Fingerprint Model:",
                       choices = available_models,
                       selected = available_models[1],
                       width = "100%"),
            sliderInput("num_predictions",
                       "Number of Top Predictions:",
                       value = 5,
                       min = 1,
                       max = 20,
                       step = 1,
                       width = "100%"),
            br(),
            actionButton("predict_btn", "Predict Targets", class = "btn-success btn-large")
          )
        )
      )
      )
    ),

    # Results Section ----
    uiOutput("results_section")
  )
)

# Server Logic ----
server <- function(input, output, session) {

  # Reactive Values ----
  rv <- reactiveValues(
    input_smiles = NULL,
    valid_smiles = NULL,
    invalid_smiles = NULL,
    smiles_source = character(0),  # Track source of each SMILES
    validation_done = FALSE,
    results_df = NULL,
    current_tab = NULL
  )

  # Disable model configuration initially
  shinyjs::disable("model_type")
  shinyjs::disable("num_predictions")
  shinyjs::disable("predict_btn")

  # Example Molecule Buttons ----
  observeEvent(input$example_caffeine, {
    updateTextAreaInput(session, "manual_smiles", value = EXAMPLE_MOLECULES$Caffeine)
    updateTabsetPanel(session, "input_tabs", selected = "Manual Input")
  })

  observeEvent(input$example_ibuprofen, {
    updateTextAreaInput(session, "manual_smiles", value = EXAMPLE_MOLECULES$Ibuprofen)
    updateTabsetPanel(session, "input_tabs", selected = "Manual Input")
  })

  observeEvent(input$example_benzene, {
    updateTextAreaInput(session, "manual_smiles", value = EXAMPLE_MOLECULES$Benzene)
    updateTabsetPanel(session, "input_tabs", selected = "Manual Input")
  })

  observeEvent(input$example_ethanol, {
    updateTextAreaInput(session, "manual_smiles", value = EXAMPLE_MOLECULES$Ethanol)
    updateTabsetPanel(session, "input_tabs", selected = "Manual Input")
  })

  observeEvent(input$example_acetic_acid, {
    updateTextAreaInput(session, "manual_smiles", value = EXAMPLE_MOLECULES$`Acetic_acid`)
    updateTabsetPanel(session, "input_tabs", selected = "Manual Input")
  })

  observeEvent(input$load_all_examples, {
    example_df <- read.csv("data/example_input.csv", stringsAsFactors = FALSE)
    smiles_text <- paste(example_df$SMILES, collapse = "\n")
    updateTextAreaInput(session, "manual_smiles", value = smiles_text)
    updateTabsetPanel(session, "input_tabs", selected = "Manual Input")
  })

  # Collect SMILES Input ----
  collect_smiles <- reactive({
    req(input$input_tabs)

    smiles_list <- NULL

    if (input$input_tabs == "Draw Molecule") {
      if (!is.null(input$drawing_smiles) && nchar(input$drawing_smiles) > 0) {
        # Return as a single-element vector, not split by characters
        smiles_list <- c(input$drawing_smiles)
      }
    } else if (input$input_tabs == "Manual Input") {
      if (!is.null(input$manual_smiles) && nchar(input$manual_smiles) > 0) {
        smiles_list <- strsplit(input$manual_smiles, "\n")[[1]]
        smiles_list <- trimws(smiles_list)
        smiles_list <- smiles_list[nchar(smiles_list) > 0]
      }
    } else if (input$input_tabs == "Upload CSV") {
      if (!is.null(input$csv_file)) {
        csv_data <- read.csv(input$csv_file$datapath, stringsAsFactors = FALSE)
        if ("SMILES" %in% colnames(csv_data)) {
          smiles_list <- csv_data$SMILES
        } else {
          showNotification("CSV file must contain 'SMILES' column", type = "error")
        }
      }
    }

    return(smiles_list)
  })

  # Validation ----
  observeEvent(input$validate_btn, {
    # First check if there's any input
    smiles_list <- collect_smiles()

    if (is.null(smiles_list) || length(smiles_list) == 0) {
      showNotification("Please enter some SMILES strings first", type = "error")
      return()
    }

    # Then check if there are previous results
    if (!is.null(rv$results_df) && nrow(rv$results_df) > 0) {
      showModal(modalDialog(
        title = "Previous Results Detected",
        "You have previous prediction results. Starting a new validation will clear these results.",
        tags$br(),
        tags$strong("Please save your results before continuing if needed."),
        footer = tagList(
          modalButton("Cancel"),
          actionButton("confirm_validate", "Continue", class = "btn-success")
        ),
        easyClose = FALSE
      ))
      return()
    }

    withProgress(message = 'Validating SMILES...', value = 0, {

      # Delete old molecule images before validation
      old_images <- list.files("www/molecules", pattern = "^mol_.*\\.png$", full.names = TRUE)
      if (length(old_images) > 0) {
        file.remove(old_images)
        cat(sprintf("✓ Deleted %d old molecule images\n", length(old_images)))
      }

      # Debug: print what we're sending
      cat("DEBUG: Number of SMILES:", length(smiles_list), "\n")
      cat("DEBUG: SMILES list:", paste(smiles_list, collapse = " | "), "\n")

      # Call Python preprocessing function
      # Explicitly convert to list for reticulate
      result <- preprocess_smiles_list(as.list(smiles_list))

      rv$valid_smiles <- result[[1]]
      rv$invalid_smiles <- result[[2]]
      rv$validation_done <- TRUE

      incProgress(1)
    })

    # Show validation results in modal dialog
    n_valid <- length(rv$valid_smiles)
    n_invalid <- length(rv$invalid_smiles)

    modal_content <- tagList(
      h4("Validation Summary"),
      div(style = "margin: 15px 0;",
        tags$p(HTML(sprintf("<span style='color: #28a745; font-weight: bold;'>✓ Valid SMILES: %d</span>", n_valid))),
        tags$p(HTML(sprintf("<span style='color: #dc3545; font-weight: bold;'>✗ Invalid SMILES: %d</span>", n_invalid)))
      ),

      if (n_valid > 0) {
        tagList(
          h5("Valid SMILES:"),
          DTOutput("valid_smiles_modal_table")
        )
      },

      if (n_invalid > 0) {
        tagList(
          br(),
          h5("Invalid SMILES:"),
          DTOutput("invalid_smiles_modal_table")
        )
      }
    )

    showModal(modalDialog(
      title = "Validation Results",
      modal_content,
      size = "l",
      easyClose = TRUE,
      footer = modalButton("Close")
    ))

    # Enable model configuration if we have valid SMILES
    if (length(rv$valid_smiles) > 0) {
      shinyjs::enable("model_type")
      shinyjs::enable("num_predictions")
      shinyjs::enable("predict_btn")
    }
  })

  # Handle confirmation to continue with validation (clearing previous results)
  observeEvent(input$confirm_validate, {
    removeModal()

    # Clear previous results
    rv$results_df <- NULL

    # Delete old molecule images
    old_images <- list.files("www/molecules", pattern = "^mol_.*\\.png$", full.names = TRUE)
    if (length(old_images) > 0) {
      file.remove(old_images)
      cat(sprintf("✓ Deleted %d old molecule images\n", length(old_images)))
    }

    smiles_list <- collect_smiles()

    if (is.null(smiles_list) || length(smiles_list) == 0) {
      showNotification("Please enter some SMILES strings first", type = "error")
      return()
    }

    withProgress(message = 'Validating SMILES...', value = 0, {

      # Debug: print what we're sending
      cat("DEBUG: Number of SMILES:", length(smiles_list), "\n")
      cat("DEBUG: SMILES list:", paste(smiles_list, collapse = " | "), "\n")

      # Call Python preprocessing function
      # Explicitly convert to list for reticulate
      result <- preprocess_smiles_list(as.list(smiles_list))

      rv$valid_smiles <- result[[1]]
      rv$invalid_smiles <- result[[2]]
      rv$validation_done <- TRUE

      incProgress(1)
    })

    # Show validation results in modal dialog
    n_valid <- length(rv$valid_smiles)
    n_invalid <- length(rv$invalid_smiles)

    modal_content <- tagList(
      h4("Validation Summary"),
      div(style = "margin: 15px 0;",
        tags$p(HTML(sprintf("<span style='color: #28a745; font-weight: bold;'>✓ Valid SMILES: %d</span>", n_valid))),
        tags$p(HTML(sprintf("<span style='color: #dc3545; font-weight: bold;'>✗ Invalid SMILES: %d</span>", n_invalid)))
      ),

      if (n_valid > 0) {
        tagList(
          h5("Valid SMILES:"),
          DTOutput("valid_smiles_modal_table")
        )
      },

      if (n_invalid > 0) {
        tagList(
          br(),
          h5("Invalid SMILES:"),
          DTOutput("invalid_smiles_modal_table")
        )
      }
    )

    showModal(modalDialog(
      title = "Validation Results",
      modal_content,
      size = "l",
      easyClose = TRUE,
      footer = modalButton("Close")
    ))

    # Enable model configuration if we have valid SMILES
    if (length(rv$valid_smiles) > 0) {
      shinyjs::enable("model_type")
      shinyjs::enable("num_predictions")
      shinyjs::enable("predict_btn")
    }
  })

  # Render Validation Panel ----
  output$validation_panel <- renderUI({
    req(rv$validation_done)

    n_valid <- length(rv$valid_smiles)
    n_invalid <- length(rv$invalid_smiles)

    div(class = "validation-panel",
      br(),
      h3("Validation Results"),
      div(class = "validation-summary",
        p(sprintf("✓ Valid SMILES: %d", n_valid), class = "validation-valid"),
        p(sprintf("✗ Invalid SMILES: %d", n_invalid), class = "validation-invalid")
      ),

      if (n_valid > 0) {
        div(
          h4("Valid SMILES:"),
          DTOutput("valid_smiles_table")
        )
      },

      if (n_invalid > 0) {
        div(
          h4("Invalid SMILES:"),
          DTOutput("invalid_smiles_table")
        )
      }
    )
  })

  output$valid_smiles_table <- renderDT({
    req(rv$valid_smiles)

    df <- data.frame(
      Index = 1:length(rv$valid_smiles),
      SMILES = rv$valid_smiles,
      Source = if(length(rv$smiles_source) >= length(rv$valid_smiles))
                  rv$smiles_source[1:length(rv$valid_smiles)]
               else
                  rep("Unknown", length(rv$valid_smiles)),
      Status = "Valid"
    )

    datatable(df,
              options = list(pageLength = 5, dom = 'tp'),
              rownames = FALSE)
  })

  output$invalid_smiles_table <- renderDT({
    req(rv$invalid_smiles)

    df <- data.frame(
      Index = 1:length(rv$invalid_smiles),
      SMILES = rv$invalid_smiles,
      Status = "Invalid"
    )

    datatable(df,
              options = list(pageLength = 5, dom = 'tp'),
              rownames = FALSE)
  })

  # Render validation tables for modal dialog ----
  output$valid_smiles_modal_table <- renderDT({
    req(rv$valid_smiles)

    df <- data.frame(
      Index = 1:length(rv$valid_smiles),
      SMILES = rv$valid_smiles,
      Source = if(length(rv$smiles_source) >= length(rv$valid_smiles))
                  rv$smiles_source[1:length(rv$valid_smiles)]
               else
                  rep("Unknown", length(rv$valid_smiles)),
      Status = "Valid"
    )

    datatable(df,
              options = list(pageLength = 10, dom = 'tp', scrollX = TRUE),
              rownames = FALSE)
  })

  output$invalid_smiles_modal_table <- renderDT({
    req(rv$invalid_smiles)

    df <- data.frame(
      Index = 1:length(rv$invalid_smiles),
      SMILES = rv$invalid_smiles,
      Status = "Invalid"
    )

    datatable(df,
              options = list(pageLength = 10, dom = 'tp', scrollX = TRUE),
              rownames = FALSE)
  })

  # Prediction ----
  observeEvent(input$predict_btn, {
    req(rv$valid_smiles)
    req(input$model_type)

    withProgress(message = 'Calculating fingerprints and predictions...', {

      # Step 1: Calculate fingerprints
      incProgress(0.2, detail = "Calculating fingerprints...")

      py_run_string(sprintf("selected_fp = FINGERPRINT_FUNCTIONS['%s']", input$model_type))
      py_run_string(sprintf("valid_smiles = %s",
                           jsonlite::toJSON(rv$valid_smiles, auto_unbox = TRUE)))
      py_run_string("fingerprints = [selected_fp(smi) for smi in valid_smiles]")
      py_run_string("import numpy as np")
      py_run_string("fp_array = np.array([list(fp) if fp is not None else [0]*4096 for fp in fingerprints])")

      # For Fused model, the fingerprint size is different (8192)
      if (input$model_type == "Fused") {
        py_run_string("fp_array = np.array([list(fp) if fp is not None else [0]*8192 for fp in fingerprints])")
      }

      # Step 2: Make predictions
      incProgress(0.4, detail = "Making predictions...")

      py_run_string(sprintf("model = MODELS_%s", input$model_type))
      py_run_string("predictions_proba = model.predict_proba(fp_array)")
      py_run_string("predictions = np.array([proba[:, 1] for proba in predictions_proba]).T")

      # Step 3: Get top predictions
      incProgress(0.6, detail = "Processing results...")

      predictions <- py$predictions

      # Generate results dataframe
      results_list <- list()

      for (i in 1:length(rv$valid_smiles)) {
        pred_scores <- predictions[i, ]
        top_indices <- order(pred_scores, decreasing = TRUE)[1:input$num_predictions]

        for (rank in 1:input$num_predictions) {
          idx <- top_indices[rank]
          target_id <- TARGET_LABELS[idx]

          results_list[[length(results_list) + 1]] <- data.frame(
            molecule_index = i,
            smiles = rv$valid_smiles[i],
            rank = rank,
            target_id = target_id,
            target_name = TARGET_ID_TO_NAME[[target_id]],
            confidence = round(pred_scores[idx], 4),
            class = TARGET_ID_TO_CLASS[[target_id]],
            type = TARGET_ID_TO_TYPE[[target_id]],
            organism = TARGET_ID_TO_ORGANISM[[target_id]],
            stringsAsFactors = FALSE
          )
        }
      }

      rv$results_df <- bind_rows(results_list)

      # Step 4: Generate molecule images
      incProgress(0.8, detail = "Generating molecule images...")

      for (i in 1:length(rv$valid_smiles)) {
        py_run_string(sprintf("from rdkit.Chem import Draw"))
        py_run_string(sprintf("from rdkit import Chem"))
        py_run_string(sprintf("mol = Chem.MolFromSmiles('%s')", rv$valid_smiles[i]))
        py_run_string(sprintf("Draw.MolToFile(mol, 'www/molecules/mol_%d.png', size=(300,200))", i))
      }

      incProgress(1, detail = "Done!")
    })

    showNotification("Predictions completed successfully!", type = "message")

    # Clear data after prediction to avoid accumulation
    rv$valid_smiles <- character(0)
    rv$invalid_smiles <- character(0)
    rv$smiles_source <- character(0)
    rv$validation_done <- FALSE
    updateTextAreaInput(session, "manual_smiles", value = "")

    # Disable predict button until next validation
    shinyjs::disable("model_type")
    shinyjs::disable("num_predictions")
    shinyjs::disable("predict_btn")
  })

  # Render Results Section ----
  output$results_section <- renderUI({
    req(rv$results_df)

    div(class = "results-section",
      br(),
      h2("Prediction Results"),

      # Summary Statistics
      div(class = "summary-cards",
        fluidRow(
          column(3,
            div(class = "summary-card",
              h4("Total Molecules"),
              h2(length(unique(rv$results_df$molecule_index)))
            )
          ),
          column(3,
            div(class = "summary-card",
              h4("Unique Targets"),
              h2(length(unique(rv$results_df$target_id)))
            )
          ),
          column(3,
            div(class = "summary-card",
              h4("Avg Confidence"),
              h2(sprintf("%.2f", mean(rv$results_df$confidence)))
            )
          ),
          column(3,
            div(class = "summary-card",
              h4("Model Used"),
              h2(input$model_type)
            )
          )
        )
      ),

      br(),

      # Statistical Analysis
      h3("Statistical Analysis"),
      fluidRow(
        column(3, plotlyOutput("confidence_by_rank_plot", height = "250px")),
        column(3, plotlyOutput("protein_class_plot", height = "250px")),
        column(3, plotlyOutput("organism_plot", height = "250px")),
        column(3, plotlyOutput("target_type_plot", height = "250px"))
      ),

      br(),

      # Detailed Predictions header with Download Buttons
      fluidRow(
        column(6,
          h3("Detailed Predictions", style = "margin-top: 0;")
        ),
        column(6,
          div(class = "download-buttons", style = "text-align: right;",
            downloadButton("download_csv", "Download CSV", class = "btn-download"),
            downloadButton("download_excel", "Download Excel", class = "btn-download"),
            downloadButton("download_top5", "Download Top-5 Summary", class = "btn-download")
          )
        )
      ),
      p(class = "help-text", style = "color: #666; font-style: italic;",
        "💡 Hover over SMILES to see molecule structure"),
      DTOutput("results_table"),

      # Force tooltip reinitialization after results are rendered
      tags$script(HTML("
        setTimeout(function() {
          if (typeof initializeMoleculeTooltips !== 'undefined') {
            console.log('Forcing tooltip reinitialization after new results...');
            initializeMoleculeTooltips();
            // Reinitialize again after a delay to ensure images are loaded
            setTimeout(initializeMoleculeTooltips, 1500);
          }
        }, 1000);
      "))
    )
  })

  # Results Table ----
  output$results_table <- renderDT({
    req(rv$results_df)

    df <- rv$results_df %>%
      mutate(
        # Add hover tooltip to SMILES column
        smiles = sprintf('<span class="smiles-hover" data-mol-id="%d">%s</span>',
                        molecule_index, smiles),
        # Keep target_id as clickable link
        target_id = sprintf('<a href="https://www.ebi.ac.uk/chembl/target_report_card/%s" target="_blank">%s</a>',
                           target_id, target_id)
      )

    datatable(df,
              escape = FALSE,
              colnames = c('Index' = 'molecule_index',
                          'Smiles' = 'smiles',
                          'Rank' = 'rank',
                          'Target ID' = 'target_id',
                          'Target Name' = 'target_name',
                          'Confidence' = 'confidence',
                          'Class' = 'class',
                          'Type' = 'type',
                          'Organism' = 'organism'),
              options = list(
                pageLength = 20,
                scrollX = TRUE,
                order = list(list(0, 'asc'), list(2, 'asc'))
              ),
              rownames = FALSE) %>%
      formatStyle('Confidence',
                  background = styleColorBar(c(0, 1), '#2ca02c'),
                  backgroundSize = '98% 88%',
                  backgroundRepeat = 'no-repeat',
                  backgroundPosition = 'center')
  })

  # Statistical Plots ----
  output$protein_class_plot <- renderPlotly({
    req(rv$results_df)

    class_counts <- rv$results_df %>%
      group_by(class) %>%
      summarise(count = n()) %>%
      arrange(desc(count))

    plot_ly(class_counts, x = ~class, y = ~count, type = 'bar',
            marker = list(color = '#08517f')) %>%
      layout(title = list(text = "Protein Class Distribution", font = list(size = 12)),
             xaxis = list(title = list(text = "Class", font = list(size = 10))),
             yaxis = list(title = list(text = "Count", font = list(size = 10))),
             margin = list(l = 40, r = 10, t = 40, b = 40))
  })

  output$organism_plot <- renderPlotly({
    req(rv$results_df)

    org_counts <- rv$results_df %>%
      group_by(organism) %>%
      summarise(count = n())

    plot_ly(org_counts, labels = ~organism, values = ~count, type = 'pie',
            textposition = 'inside', textinfo = 'percent',
            marker = list(line = list(color = '#FFFFFF', width = 1))) %>%
      layout(title = list(text = "Organism Distribution", font = list(size = 12)),
             margin = list(l = 10, r = 10, t = 40, b = 10),
             showlegend = TRUE,
             legend = list(orientation = 'v', x = 1, y = 0.5, font = list(size = 9)))
  })

  output$target_type_plot <- renderPlotly({
    req(rv$results_df)

    type_counts <- rv$results_df %>%
      group_by(type) %>%
      summarise(count = n())

    plot_ly(type_counts, labels = ~type, values = ~count, type = 'pie',
            textposition = 'inside', textinfo = 'percent',
            marker = list(line = list(color = '#FFFFFF', width = 1))) %>%
      layout(title = list(text = "Target Type Distribution", font = list(size = 12)),
             margin = list(l = 10, r = 10, t = 40, b = 10),
             showlegend = TRUE,
             legend = list(orientation = 'v', x = 1, y = 0.5, font = list(size = 9)))
  })

  output$confidence_by_rank_plot <- renderPlotly({
    req(rv$results_df)

    conf_by_rank <- rv$results_df %>%
      group_by(rank) %>%
      summarise(
        mean_conf = mean(confidence),
        sd_conf = sd(confidence)
      )

    plot_ly(conf_by_rank, x = ~rank, y = ~mean_conf, type = 'scatter', mode = 'lines+markers',
            marker = list(color = '#08517f', size = 8),
            line = list(color = '#08517f', width = 2)) %>%
      layout(title = list(text = "Average Confidence by Rank", font = list(size = 12)),
             xaxis = list(title = list(text = "Rank", font = list(size = 10))),
             yaxis = list(title = list(text = "Mean Confidence Score", font = list(size = 10))),
             margin = list(l = 50, r = 10, t = 40, b = 40))
  })

  # Download Handlers ----
  output$download_csv <- downloadHandler(
    filename = function() {
      sprintf("predictions_%s_%s.csv", input$model_type, Sys.Date())
    },
    content = function(file) {
      write.csv(rv$results_df, file, row.names = FALSE)
    }
  )

  output$download_excel <- downloadHandler(
    filename = function() {
      sprintf("predictions_%s_%s.xlsx", input$model_type, Sys.Date())
    },
    content = function(file) {
      write_xlsx(rv$results_df, file)
    }
  )

  output$download_top5 <- downloadHandler(
    filename = function() {
      sprintf("top5_summary_%s_%s.csv", input$model_type, Sys.Date())
    },
    content = function(file) {
      top5_df <- rv$results_df %>%
        filter(rank <= 5)
      write.csv(top5_df, file, row.names = FALSE)
    }
  )
}

# Run App ----
shinyApp(ui = ui, server = server)
