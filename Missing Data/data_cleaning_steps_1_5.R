#===============================================================================
# DATA CLEANING AND PREPROCESSING: STEPS 1-5
# Course: Data Cleaning and Handling Missing Data using R and MICE
#===============================================================================

# Clear environment
rm(list = ls())

#-------------------------------------------------------------------------------
# SETUP: Load Required Packages
#-------------------------------------------------------------------------------
# Install packages if needed (uncomment as necessary)
# install.packages("dplyr")
# install.packages("tidyverse")
# install.packages("mice")
# install.packages("lubridate")

library(dplyr)
library(tidyverse)
library(lubridate)

#-------------------------------------------------------------------------------
# STEP 0: SET WORKING DIRECTORY AND LOAD DATA
#-------------------------------------------------------------------------------
# Set your working directory to where your data files are located
# setwd("YOUR/PATH/HERE")  # Modify this to your project folder

# Initialize variables to track excluded columns and rows
excluded_cols <- c()
excluded_rows <- c()

# Load the data files
MainDF <- read.csv("Raw.csv")
StoreDF <- read.csv("StoreTable.csv")
ConcessDF <- read.csv("ConcessTable.csv")
CustomerDF <- read.csv("CustomerTable.csv")

# Initial data exploration
cat("=== INITIAL DATA EXPLORATION ===\n\n")

cat("MainDF dimensions:", dim(MainDF)[1], "rows x", dim(MainDF)[2], "columns\n")
cat("StoreDF dimensions:", dim(StoreDF)[1], "rows x", dim(StoreDF)[2], "columns\n")
cat("ConcessDF dimensions:", dim(ConcessDF)[1], "rows x", dim(ConcessDF)[2], "columns\n")
cat("CustomerDF dimensions:", dim(CustomerDF)[1], "rows x", dim(CustomerDF)[2], "columns\n\n")

cat("Missing values per column:\n")
cat("\nMainDF:\n")
print(colSums(is.na(MainDF)))
cat("\nStoreDF:\n")
print(colSums(is.na(StoreDF)))
cat("\nConcessDF:\n")
print(colSums(is.na(ConcessDF)))
cat("\nCustomerDF:\n")
print(colSums(is.na(CustomerDF)))

#-------------------------------------------------------------------------------
# DATA WRANGLING: Create Flat File (FFdf)
#-------------------------------------------------------------------------------
cat("\n=== DATA WRANGLING: CREATING FLAT FILE ===\n\n")

# Start with MainDF as the base
FFdf <- MainDF
cat("Starting rows:", nrow(FFdf), "\n")

# Merge with StoreDF (left join on Cust_ID)
FFdf <- merge(FFdf, StoreDF, by = "Cust_ID", all.x = TRUE)
cat("After merging StoreDF:", nrow(FFdf), "rows\n")

# Merge with ConcessDF (left join on Cust_ID)
FFdf <- merge(FFdf, ConcessDF, by = "Cust_ID", all.x = TRUE)
cat("After merging ConcessDF:", nrow(FFdf), "rows\n")

# Merge with CustomerDF (left join on Cust_ID)
FFdf <- merge(FFdf, CustomerDF, by = "Cust_ID", all.x = TRUE)
cat("After merging CustomerDF:", nrow(FFdf), "rows\n")

# Store dimensions for reporting
flat_file_dims <- dim(FFdf)
cat("\nFinal flat file dimensions:", flat_file_dims[1], "rows x", flat_file_dims[2], "columns\n")

#===============================================================================
# STEP 1: OPEN DATA AND ARRANGE COLUMNS
#===============================================================================
cat("\n=== STEP 1: OPEN DATA AND ARRANGE COLUMNS ===\n\n")

# 1a. Create a unique identifier variable (row number)
FFdf$ID <- 1:nrow(FFdf)
cat("Created ID variable (1 to", nrow(FFdf), ")\n")

# 1b. Reorder columns: Y-variable first, then alphabetically, with ID
# Identify the Y-variable (outcome/dependent variable)
y_var <- "Y01"  # Modify this if your Y-variable has a different name

# Check if Y-variable exists
if (y_var %in% names(FFdf)) {
    other_vars <- sort(setdiff(names(FFdf), y_var))
    FFdf <- FFdf[, c(y_var, other_vars)]
    cat("Reordered columns with", y_var, "first, then alphabetically\n")
} else {
    cat("Warning: Y-variable '", y_var, "' not found. Columns ordered alphabetically.\n")
    FFdf <- FFdf[, sort(names(FFdf))]
}

# 1c. Check the structure of the dataset
cat("\nDataset Structure:\n")
str(FFdf)

# 1d. Summary of numerical variables to spot outliers
cat("\nSummary Statistics:\n")
print(summary(FFdf))

# 1e. Check variable classes
cat("\nVariable Classes:\n")
print(sapply(FFdf, class))

# 1f. Visual inspection of numerical variables (histograms)
# Identify numeric columns for plotting
numeric_cols <- names(FFdf)[sapply(FFdf, is.numeric)]
cat("\nNumeric columns available for histograms:", paste(numeric_cols, collapse = ", "), "\n")

# Example: Create histogram for a numeric variable (uncomment and modify as needed)
# hist(FFdf$Weekend.Attended, main = "Distribution of Weekend Attended", 
#      xlab = "Weekend Attended", col = "lightblue")

#===============================================================================
# STEP 2: REVIEW VARIABLES FOR COMMON SENSE (SME KNOWLEDGE)
#===============================================================================
cat("\n=== STEP 2: REVIEW VARIABLES FOR COMMON SENSE ===\n\n")

# 2a. Clean up variable names - replace spaces and periods with underscores
names(FFdf) <- gsub("\\.", "_", names(FFdf))
names(FFdf) <- gsub(" ", "_", names(FFdf))
cat("Replaced periods and spaces with underscores in variable names\n")

# 2b. Capitalize first letter of each word in variable names
names(FFdf) <- gsub("(^|_)([a-z])", "\\1\\U\\2", names(FFdf), perl = TRUE)
cat("Capitalized first letter of each word in variable names\n")

# Display cleaned variable names
cat("\nCleaned Variable Names:\n")
print(names(FFdf))

# 2c. Check for duplicate IDs or linking variables
cat("\nChecking for duplicate Cust_IDs:\n")
n_unique_cust <- length(unique(FFdf$Cust_ID))
n_total <- nrow(FFdf)
cat("Unique Cust_IDs:", n_unique_cust, "\n")
cat("Total rows:", n_total, "\n")
if (n_unique_cust != n_total) {
    cat("WARNING: There are duplicate Cust_IDs! Consider investigating.\n")
    # Find duplicates
    dup_ids <- FFdf$Cust_ID[duplicated(FFdf$Cust_ID)]
    cat("Duplicate IDs:", paste(unique(dup_ids), collapse = ", "), "\n")
} else {
    cat("All Cust_IDs are unique.\n")
}

# 2d. Common sense checks - review ranges and values
cat("\nCommon Sense Checks:\n")
cat("Review the summary statistics above for any impossible or implausible values.\n")
cat("Examples to look for:\n")
cat("  - Negative values where only positive expected\n")
cat("  - Ages outside reasonable range (e.g., < 0 or > 120)\n")
cat("  - Percentages outside 0-100 range\n")
cat("  - Dates in the future or too far in the past\n")

#===============================================================================
# STEP 3: REVIEW/FIX VARIABLE CODING (Nominal, Continuous, Ordinal)
#===============================================================================
cat("\n=== STEP 3: REVIEW AND FIX VARIABLE CODING ===\n\n")

# 3a. Identify character variables and convert to factors
chr_vars <- names(FFdf)[sapply(FFdf, is.character)]
cat("Character variables found:", paste(chr_vars, collapse = ", "), "\n")

if (length(chr_vars) > 0) {
    for (var in chr_vars) {
        FFdf[[var]] <- as.factor(FFdf[[var]])
        cat("Converted", var, "to factor with", length(levels(FFdf[[var]])), "levels\n")
    }
}

# 3b. Review factor levels
cat("\nFactor Variables and Their Levels:\n")
factor_vars <- names(FFdf)[sapply(FFdf, is.factor)]
for (var in factor_vars) {
    cat("\n", var, ":\n")
    print(table(FFdf[[var]], useNA = "ifany"))
}

# 3c. Handle ordinal variables (example with Educational_Level if it exists)
# Ordinal variables have a natural order (e.g., Low < Medium < High)
if ("Educational_Level" %in% names(FFdf)) {
    cat("\nConverting Educational_Level to ordered factor:\n")
    # Define the order (modify based on your actual levels)
    edu_levels <- c("Less than High School", "High School", "Some College", 
                    "Bachelor's Degree", "Master's Degree", "Doctorate")
    # Check which levels actually exist
    actual_levels <- intersect(edu_levels, levels(FFdf$Educational_Level))
    if (length(actual_levels) > 0) {
        FFdf$Educational_Level <- factor(FFdf$Educational_Level, 
                                          levels = actual_levels, 
                                          ordered = TRUE)
        cat("Educational_Level converted to ordered factor\n")
    } else {
        cat("Note: Educational_Level levels don't match predefined order. Review levels:\n")
        print(levels(FFdf$Educational_Level))
    }
}

# 3d. Display final variable classes
cat("\nFinal Variable Classes:\n")
print(sapply(FFdf, class))

# 3e. Create a backup before further modifications
FFdf_backup <- FFdf
cat("\nBackup created: FFdf_backup\n")

#===============================================================================
# STEP 4: DATA INTEGRITY AND VALIDATION CHECKS
#===============================================================================
cat("\n=== STEP 4: DATA INTEGRITY AND VALIDATION CHECKS ===\n\n")

# 4a. Check for misspelled factor levels
cat("Reviewing factor levels for potential misspellings:\n")
for (var in factor_vars) {
    lvls <- levels(FFdf[[var]])
    cat("\n", var, "levels:\n")
    print(lvls)
    # Look for similar levels that might be misspellings
    if (length(lvls) > 1) {
        for (i in 1:(length(lvls)-1)) {
            for (j in (i+1):length(lvls)) {
                # Simple similarity check (could use stringdist package for more sophisticated)
                if (tolower(lvls[i]) == tolower(lvls[j])) {
                    cat("  WARNING: Possible duplicate levels (case difference):", lvls[i], "vs", lvls[j], "\n")
                }
            }
        }
    }
}

# 4b. Check for bogus/impossible values in numeric variables
cat("\nChecking numeric variables for impossible values:\n")
for (var in numeric_cols) {
    if (var %in% names(FFdf)) {
        var_range <- range(FFdf[[var]], na.rm = TRUE)
        cat("\n", var, ": min =", var_range[1], ", max =", var_range[2], "\n")
        
        # Check for negative values (flag if unexpected)
        if (var_range[1] < 0) {
            cat("  Note: Contains negative values\n")
        }
    }
}

# 4c. Function to combine similar factor levels
combine_levels <- function(df, var_name, old_levels, new_level) {
    if (var_name %in% names(df) && is.factor(df[[var_name]])) {
        levels(df[[var_name]])[levels(df[[var_name]]) %in% old_levels] <- new_level
        cat("Combined levels in", var_name, ":", paste(old_levels, collapse = ", "), "->", new_level, "\n")
    }
    return(df)
}

# Example: Combine similar levels (uncomment and modify as needed)
# FFdf <- combine_levels(FFdf, "Gender", c("M", "Male", "male"), "Male")
# FFdf <- combine_levels(FFdf, "Gender", c("F", "Female", "female"), "Female")

# 4d. Remove problematic rows if identified
# Example: Remove row with specific ID (uncomment and modify as needed)
# problem_row <- 66723  # Row with no valid ID
# if (problem_row %in% FFdf$ID) {
#     excluded_rows <- c(excluded_rows, problem_row)
#     FFdf <- FFdf[FFdf$ID != problem_row, ]
#     cat("Removed row with ID:", problem_row, "\n")
# }

# 4e. Document any excluded columns
# Example: Exclude columns that are not useful for analysis
# cols_to_exclude <- c("Unused_Column1", "Unused_Column2")
# excluded_cols <- c(excluded_cols, cols_to_exclude)
# FFdf <- FFdf[, !(names(FFdf) %in% cols_to_exclude)]

cat("\nExcluded columns:", ifelse(length(excluded_cols) == 0, "None", paste(excluded_cols, collapse = ", ")), "\n")
cat("Excluded rows:", ifelse(length(excluded_rows) == 0, "None", paste(excluded_rows, collapse = ", ")), "\n")

#===============================================================================
# STEP 5: HANDLE DATES
#===============================================================================
cat("\n=== STEP 5: HANDLE DATES ===\n\n")

# 5a. Identify potential date columns
# Look for columns with "date", "time", "dt" in their names
potential_date_cols <- names(FFdf)[grepl("date|time|dt|Date|Time|DT", names(FFdf), ignore.case = TRUE)]
cat("Potential date columns identified:", 
    ifelse(length(potential_date_cols) == 0, "None", paste(potential_date_cols, collapse = ", ")), "\n")

# 5b. Function to convert and extract date components
process_date_column <- function(df, date_col, date_format = "%Y-%m-%d") {
    if (date_col %in% names(df)) {
        # Convert to Date type
        df[[date_col]] <- as.Date(df[[date_col]], format = date_format)
        
        # Extract components
        df[[paste0(date_col, "_Year")]] <- year(df[[date_col]])
        df[[paste0(date_col, "_Month")]] <- month(df[[date_col]])
        df[[paste0(date_col, "_Day")]] <- day(df[[date_col]])
        df[[paste0(date_col, "_DayOfWeek")]] <- wday(df[[date_col]], label = TRUE)
        df[[paste0(date_col, "_Quarter")]] <- quarter(df[[date_col]])
        
        cat("Processed date column:", date_col, "\n")
        cat("  Created:", paste0(date_col, "_Year"), ",", 
            paste0(date_col, "_Month"), ",",
            paste0(date_col, "_Day"), ",",
            paste0(date_col, "_DayOfWeek"), ",",
            paste0(date_col, "_Quarter"), "\n")
    }
    return(df)
}

# 5c. Process identified date columns (uncomment and modify format as needed)
# Common date formats:
#   "%Y-%m-%d"     -> 2024-01-15
#   "%m/%d/%Y"     -> 01/15/2024
#   "%d-%m-%Y"     -> 15-01-2024
#   "%Y/%m/%d"     -> 2024/01/15

# Example: Process a date column
# FFdf <- process_date_column(FFdf, "Transaction_Date", date_format = "%Y-%m-%d")

# 5d. Calculate time-based features (if applicable)
# Example: Days since a reference date
# reference_date <- as.Date("2024-01-01")
# FFdf$Days_Since_Reference <- as.numeric(FFdf$Transaction_Date - reference_date)

# Example: Time between two dates
# FFdf$Days_Between <- as.numeric(FFdf$End_Date - FFdf$Start_Date)

#===============================================================================
# FINAL SUMMARY
#===============================================================================
cat("\n=== FINAL SUMMARY ===\n\n")

cat("Final dataset dimensions:", dim(FFdf)[1], "rows x", dim(FFdf)[2], "columns\n\n")

cat("Variable types summary:\n")
var_types <- sapply(FFdf, class)
type_summary <- table(sapply(var_types, function(x) x[1]))
print(type_summary)

cat("\nMissing values summary:\n")
missing_summary <- colSums(is.na(FFdf))
missing_summary <- missing_summary[missing_summary > 0]
if (length(missing_summary) > 0) {
    print(missing_summary)
} else {
    cat("No missing values in the dataset.\n")
}

cat("\nColumn names:\n")
print(names(FFdf))

cat("\nExcluded columns:", ifelse(length(excluded_cols) == 0, "None", paste(excluded_cols, collapse = ", ")), "\n")
cat("Excluded rows:", ifelse(length(excluded_rows) == 0, "None", paste(excluded_rows, collapse = ", ")), "\n")

# Save the cleaned dataset
# write.csv(FFdf, "Cleaned_Data.csv", row.names = FALSE)
# cat("\nCleaned data saved to 'Cleaned_Data.csv'\n")

cat("\n=== DATA CLEANING STEPS 1-5 COMPLETE ===\n")
cat("Next steps: Proceed to missing data analysis and imputation with MICE package.\n")
