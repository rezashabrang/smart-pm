### Data Cleaning and Preprocessing from DSLIB Files

---

### **Task Objective**

The goal of this task is to gain practical skills of:

1. Identifying and validating usable data from project management datasets.
2. Extracting meaningful data from multiple sheets.
3. Applying data cleaning and preprocessing techniques to prepare the data for machine learning.

---

### **Instructions**

#### **Step 1: File and Data Validation**

- **Choose a File:** Select a DSLIB file from the provided library.
- **Sheet Review:** Examine all sheets in the file and identify one or more sheets with usable data.
- **Validation Criteria:**
  - Data relevance: Does the sheet provide meaningful information for project management (e.g., schedules, resources, risks)?
  - Data quality: Are the key columns complete or repairable with cleaning methods?

#### **Step 2: Data Extraction**

- **Select Columns:** Choose columns relevant to project management analysis. For example:
  - **Baseline Schedule:** Task IDs, durations, and relationships.
  - **Resources:** Resource names, demand, and costs.
  - **Risk Analysis:** Risk profiles and activity durations.
  - **Time-Series Data:** Use data from sheets like "TP1," "TP2," etc., for time-bound tasks.
- **Extract Data:** Load the data from the chosen sheet(s) and prepare it for cleaning.

#### **Step 3: Data Cleaning**

Perform the following cleaning tasks:

1. **Missing Values:**
   - Impute missing numerical values using mean/median or remove rows/columns with excessive missingness.
   - Handle missing categorical values by imputing the mode.
2. **Duplicates:**
   - Identify and remove duplicate rows to maintain data integrity.
3. **Outliers:**
   - Detect outliers in numerical columns using statistical methods like Z-scores or IQR.
   - Decide whether to remove or adjust the outliers.
4. **Unstructured Data:**
   - If you choose a sheet with text data (e.g., "Risk Analysis"), apply text-cleaning techniques:
     - Tokenization.
     - Removing stop words.
     - Lemmatization or stemming.

#### **Step 4: Data Preprocessing**

- **Feature Scaling:** Apply one of the following techniques to numerical data:
  - Min-Max Scaling: Normalize data to a range of 0 to 1.
  - Standardization: Scale data using Z-scores (mean = 0, standard deviation = 1).
- **Encoding Categorical Variables:** Convert text-based categories into numerical form using:
  - One-Hot Encoding: Create binary columns for each category.
  - Label Encoding: Assign unique integers to categories.

#### **Step 5: Reporting**

Prepare a short report summarizing:

1. **Data Selection and Validation:** Which file and sheet(s) were chosen? Why are they relevant?
2. **Cleaning Process:** Steps taken to clean missing values, duplicates, and outliers.
3. **Preprocessing Steps:** Scaling, encoding, and other techniques applied.
4. **Insights:** Highlight any interesting observations or patterns in the cleaned data.

#### **Deliverables**

1. A cleaned and preprocessed dataset (as an Excel or CSV file).
2. A 500–700 word report covering all steps taken, justifications for methods used, and key observations.

---

### **Evaluation Criteria**

1. **Data Selection:** Relevance and completeness of the chosen data.
2. **Cleaning Effectiveness:** The thoroughness and accuracy of cleaning steps.
3. **Preprocessing:** Correct application of scaling and encoding methods.
4. **Reporting Quality:** Clarity, depth, and structure of the report.

---
