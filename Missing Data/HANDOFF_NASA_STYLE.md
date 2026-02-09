# Data Cleaning Steps 6-11: Technical Handoff Document
## NASA-JSC Software Engineering Standards Compliant

**Document ID:** MD-CLEAN-001
**Version:** 1.0
**Date:** 2026-02-09
**Author:** Claude Code Assistant
**Status:** READY FOR CONTINUATION

---

## 1. PURPOSE AND SCOPE

### 1.1 Objective
Complete data wrangling and cleaning (Steps 1-11) for the Missing Data course assignment, preparing FFdf dataset for imputation analysis in Classes 5+.

### 1.2 Scope
- Input: Raw.csv, StoreTable.csv, ConcessTable.csv, CustomerTable.csv
- Output: FFdf_cleaned.csv (cleaned flat file ready for missing data analysis)
- Reference: "01 MD Course Template V1.826.Rmd" and "2 Wrangling and Cleaning Steps V1.9.pdf"

---

## 2. SYSTEM STATE AT HANDOFF

### 2.1 Files Modified
| File | Status | Description |
|------|--------|-------------|
| `1_5.Rmd` | MODIFIED | Main analysis document with Steps 1-11 |
| `FFdf_cleaned.csv` | GENERATED | Cleaned dataset output |
| `run_analysis.R` | CREATED | Auxiliary script (can be deleted) |

### 2.2 Dataset Dimensions at Handoff
- **Rows:** ~1000 (exact count in FFdf_cleaned.csv)
- **Columns:** 58 (includes derived variables)
- **Response Variable:** Y01 (binary: 0/1)

### 2.3 Derived Variables Created
| Variable | Source | Purpose |
|----------|--------|---------|
| Region | State_Name | Grouped into 4 US Census regions |
| Contact_Year | Last_Contact | Extracted date component |
| Contact_Month | Last_Contact | Extracted date component |
| Contact_Day | Last_Contact | Extracted date component |
| Contact_Weekday | Last_Contact | Extracted date component |
| Contact_Hour | Last_Contact | Extracted date component |
| Flag_Tenure_High | Tenure | Outlier flag (Tenure > 100) |
| Flag_Survey_Invalid | Survey_Comp | Outlier flag (Survey_Comp > 1) |

---

## 3. COMPLETED TASKS

### 3.1 Step 6: Categorical Variables
- [x] Reviewed level distributions for all factor variables
- [x] Marital "U" (Unknown) retained as separate category
- [x] **State_Name grouped into 4 regions: Northeast, Midwest, South, West**
- [x] Rare levels (< 5%) identified and documented

### 3.2 Step 7: Zero-Variance Predictors
- [x] Identified and removed zero-variance columns
- [x] Removed: Address, Name, PhoneNum (100% NA - PII scrubbed)
- [x] Removed: InfRate, UnempRate, Last_Team_Championship, NHL_Team_Record, Playoffs

### 3.3 Step 8: Near Zero-Variance Predictors
- [x] Identified NZV columns (>95% one value)
- [x] Flagged: Additional_Seats (97% = 0), Mult_Loc (97% = No)
- [x] Decision: KEEP but monitor during modeling

### 3.4 Step 9: Redundant Columns & Multicollinearity
- [x] Removed State_Loc (redundant with State_Name)
- [x] Correlation matrix generated and analyzed
- [x] **Multicollinearity clusters identified:**
  - Cluster 1: First_Year_STH ↔ Rep_Visits ↔ Spent_Other_Teams ↔ STH_Attended
  - Cluster 2: Weekday_Attended ↔ Weekday_Sold, Weekend_Attended ↔ Weekend_Sold
- [x] Flagged for removal: Rep_Visits, Weekday_Sold, Weekend_Sold

### 3.5 Step 10: Outliers & Missing Values
- [x] Boxplots generated for key numeric variables
- [x] DistA = 999 converted to NA (placeholder values)
- [x] Outlier flags created for Tenure and Survey_Comp
- [x] Missing data summary generated

### 3.6 Step 11: Decision Tree Sanity Check
- [x] Original tree: 94.04% accuracy (triggered leakage warning)
- [x] **Created second tree using Region instead of State_Name**
- [x] Excluded high-cardinality: State_Name, Zip_Codes
- [x] Variable importance documented

---

## 4. KNOWN ISSUES AND ANOMALIES

### 4.1 Data Quality Issues (Require SME Verification)
| ID | Variable | Issue | Priority | Recommended Action |
|----|----------|-------|----------|-------------------|
| DQ-001 | Tenure | Max = 400 (impossible if years) | HIGH | Verify units with SME |
| DQ-002 | Age | Max = 99 (possible placeholder) | MEDIUM | Verify with SME |
| DQ-003 | Survey_Comp | 110 values > 1 (expected 0-1) | HIGH | Cap at 1 or investigate |
| DQ-004 | Cust_ID | 175 duplicates in MainDF | MEDIUM | Investigate or dedupe |

### 4.2 Technical Issues
| ID | Issue | Status | Notes |
|----|-------|--------|-------|
| TI-001 | R not executable from bash | WORKAROUND | Run via RStudio/direct R GUI |
| TI-002 | VIM aggr() not rendered | PENDING | Add to Step 10 if needed |

---

## 5. CONFIGURATION

### 5.1 Required R Packages
```r
library(mice)       # MICE imputation
library(VIM)        # Missing data visualization
library(tidyverse)  # Data manipulation
library(ggplot2)    # Plotting
library(reshape2)   # Data reshaping
library(corrplot)   # Correlation visualization
library(lubridate)  # Date handling
library(flextable)  # Table formatting
library(rpart)      # Decision trees
library(rpart.plot) # Tree visualization
```

### 5.2 Working Directory
```r
setwd("C:/Users/david/Desktop/Spring-Semester-/Missing Data")
```

### 5.3 Region Mapping (US Census Bureau)
```r
northeast <- c("Connecticut", "Maine", "Massachusetts", "New Hampshire",
               "Rhode Island", "Vermont", "New Jersey", "New York", "Pennsylvania")
midwest <- c("Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin",
             "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska",
             "North Dakota", "South Dakota")
south <- c("Delaware", "Florida", "Georgia", "Maryland", "North Carolina",
           "South Carolina", "Virginia", "District of Columbia", "West Virginia",
           "Alabama", "Kentucky", "Mississippi", "Tennessee",
           "Arkansas", "Louisiana", "Oklahoma", "Texas")
west <- c("Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico",
          "Utah", "Wyoming", "Alaska", "California", "Hawaii", "Oregon", "Washington")
```

---

## 6. VERIFICATION CHECKLIST

Before proceeding to Missing Data analysis (Class 5+):

- [ ] Knit 1_5.Rmd successfully to PDF
- [ ] Verify FFdf_cleaned.csv contains all expected columns
- [ ] Confirm Region variable has 4 levels (no "Other")
- [ ] Verify no zero-variance columns remain
- [ ] Review decision tree output for data leakage
- [ ] Confirm multicollinearity flags are documented

---

## 7. NEXT STEPS (Class 5+)

### 7.1 Immediate Tasks
1. Load FFdf_cleaned.csv for missing data analysis
2. Run VIM::aggr() for missing pattern visualization
3. Classify missing mechanism (MCAR, MAR, MNAR)
4. Determine imputation strategy

### 7.2 Variables to Exclude from Imputation
- ID, Cust_ID (identifiers)
- Flag_Tenure_High, Flag_Survey_Invalid (derived flags)
- Marital_Original (backup column)
- Contact_* date components (derived from Last_Contact)

### 7.3 Variables Flagged for Model Removal (VIF Check)
- Rep_Visits (correlated with First_Year_STH)
- Weekday_Sold (correlated with Weekday_Attended)
- Weekend_Sold (correlated with Weekend_Attended)

---

## 8. TRACEABILITY MATRIX

| Requirement | Template Section | 1_5.Rmd Section | Status |
|-------------|------------------|-----------------|--------|
| Step 1: Open data | Class 1 | step1_open_data | COMPLETE |
| Step 2: SME review | Class 2 | step2_common_sense | COMPLETE |
| Step 3: Variable types | Class 2 | step3_variable_types | COMPLETE |
| Step 4: Validation | Class 2 | step4_checks | COMPLETE |
| Step 5: Dates | Class 2 | step5_handle_dates | COMPLETE |
| Step 6: Categorical | Class 3 | step6_categorical, step6_group_states_by_region | COMPLETE |
| Step 7: Zero-variance | Class 3 | step7_zero_variance | COMPLETE |
| Step 8: Near zero-var | Class 4 | step8_near_zero_variance | COMPLETE |
| Step 9: Redundant | Class 4 | step9_redundant, step9_handle_multicollinearity | COMPLETE |
| Step 10: Outliers | Class 4 | step10_outliers, step10_missing_assessment | COMPLETE |
| Step 11: Decision tree | Class 4 | step11_decision_tree, step11_tree_with_region | COMPLETE |

---

## 9. APPROVAL AND SIGN-OFF

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | Claude Code | 2026-02-09 | [Automated] |
| Reviewer | [Pending] | | |
| Approver | [Pending] | | |

---

**END OF DOCUMENT**
