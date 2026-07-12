## Exploratory Data Analysis (EDA)
EDA stands for Exploratory Data Analysis — it's the step where you explore the data to:
- Understand it
- Discover patterns
- Spot anomalies
- Generate insights
- Decide what to do next

-- Steps of EDA :
1) Viewing the Data :
Use `head()`, `tail()`, `shape()`, and `info()` to inspect the dataset.
Identify the available columns and their data types.
2) Summary Statistics :
Calculate mean, median, mode, standard deviation, minimum, maximum, and quartiles.
Understand the central tendency and spread of the data.
3) Value Counts :
Count the number of unique values in a column.
Useful for analyzing categorical features.
4) Missing Value Analysis :
Identify missing values in the dataset.
Calculate the percentage of missing data in each column.
5) Visualizations :
Histograms → Show the distribution of values.
Boxplots → Detect outliers and visualize data spread.
Bar Plots → Compare different categories.
Correlation Heatmaps → Show relationships between numerical features.
Scatter Plots → Visualize relationships between two variables.
6) Target Variable Exploration :
Analyze how the target (output) variable relates to other features.
Identify which variables have the strongest influence on the target.

-- EDA Importance :
You can't clean or preprocess data that you don't understand.
Helps identify mistakes, biases, outliers, and limitations in the data.
Guides the direction of data cleaning and feature engineering.
Provides a clear understanding of the data for stakeholders and decision-making.
Improves the quality of data before building machine learning models.
Leads to better model performance and more reliable predictions.


## Data Cleaning
1. Handle Missing Values
Check which columns have missing values (nulls)
Strategies to handle:
* Drop missing rows/columns (only if very few)
* Impute with:
  * Mean/Median → for numerical data
  * Mode → for categorical data
  * Advanced: Linear regression, KNN, or interpolation (for future learning)
2. Remove Duplicates
Detect and drop exact duplicate rows
3. Fix Data Types
Convert wrong types (e.g., numbers stored as strings, dates as text)
4. Handle Inconsistent Categories
Clean up categorical values like:
  * "Male", "male", "MALE" → should all become "male"
  * "Yes", "yes", "Y" → unify to one format
5. Detect and Handle Outliers
Use boxplots, IQR, or Z-score
Handle by:
* Removing (if clearly wrong)
* Capping (e.g., to 95th percentile)
6. Fix Logic or Domain Errors
E.g., age = -5 is invalid, or BMI = 200 likely an error
Can replace with mean, median, or remove
-- EDA tells you what's wrong. Data Cleaning fixes it.
-- Cleaning is not glamorous, but it's 80% of the work in real-world projects.


## Data Preprocessing
Used to prepare clean data so it can be analyzed or used in a machine learning model.
If Data Cleaning is about fixing mistakes, Data Preprocessing is about transforming valid data into a usable format.
1. Encoding Categorical Variables
Convert text labels (like "male", "yes", "southeast") into numbers.
Two common methods:
- Label Encoding (Ordinal):
  Good for ordered categories like "Low", "Medium", "High"
- One-Hot Encoding (Nominal):
  For non-ordered categories like region
2. Feature Transformation (Log, Square root, etc.)
Used to handle skewed data, like right-skewed or left skewed data.
3. Feature Scaling (Normalization or Standardization)
Bring numerical values to the same scale — especially useful for distance-based algorithms.
-- Normalization (Min-Max Scaling)
Normalization is a feature scaling technique used to bring all numerical values into the same range (usually 0 to 1).
This prevents features with larger values from dominating those with smaller values.
-- Standardization (Z-score Scaling)
Standardization is a feature scaling technique that transforms numerical data so that it has Mean = 0, Standard Deviation = 1.
Unlike normalization, it does not limit values between 0 and 1. The transformed values can be negative or greater than 1.


## Feature Engineering
Creating new features or transforming existing ones to expose useful patterns that ML models can learn from.
Why do we need it?
Because ML models don't know domain logic — we have to give them the right signals.
Common Feature Engineering Techniques in ML:
- Mathematical Combinations
- Target-Based Flags
- Binning (when it helps)
- Time-Based Features (if time exists)


## Feature Selection
Selecting the most useful features and removing the rest.
Why is it important?
- Reduces noise and overfitting
- Speeds up training
- Improves model accuracy
- Makes model interpretation easier

Common Feature Selection Techniques in ML:
1. Filter Methods (Pure Statistics)
- Correlation Matrix → Remove highly correlated features
- Chi-square test (categorical vs categorical)
- ANOVA F-test (numerical vs categorical target)
2. Embedded Methods (Selection built into the model)
- Lasso Regression → Shrinks coefficients to 0
- Tree-based models (Random Forest, XGBoost) → Feature importance scores
