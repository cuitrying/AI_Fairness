import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# Set style for plots
plt.style.use('seaborn')
sns.set_palette("husl")

# Read the main COMPAS dataset
df = pd.read_csv('compas-scores-two-years.csv')

# Convert jail dates to datetime
df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])

# Calculate length of stay in days
df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.total_seconds() / (24 * 60 * 60)

# Update columns of interest to include length_of_stay
columns_of_interest = [
    'age', 'sex', 'race', 'priors_count', 'c_charge_degree',
    'score_text', 'two_year_recid', 'length_of_stay'
]
df = df[columns_of_interest]
df = df.dropna()

# Basic Visualization Functions
def plot_recidivism_by_race():
    plt.figure(figsize=(10, 6))
    sns.barplot(x='race', y='two_year_recid', data=df)
    plt.title('Recidivism Rate by Race')
    plt.xticks(rotation=45)
    plt.ylabel('Recidivism Rate')
    plt.tight_layout()
    plt.show()

def plot_recidivism_by_age():
    plt.figure(figsize=(10, 6))
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 25, 35, 45, 55, 100], 
                            labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    sns.barplot(x='age_group', y='two_year_recid', data=df)
    plt.title('Recidivism Rate by Age Group')
    plt.ylabel('Recidivism Rate')
    plt.tight_layout()
    plt.show()
    
    print("\nRecidivism rates by age group:")
    print(df.groupby('age_group')['two_year_recid'].agg(['count', 'mean']))

def plot_recidivism_by_sex():
    plt.figure(figsize=(8, 6))
    sns.barplot(x='sex', y='two_year_recid', data=df)
    plt.title('Recidivism Rate by Sex')
    plt.ylabel('Recidivism Rate')
    plt.tight_layout()
    plt.show()

def plot_recidivism_by_stay():
    plt.figure(figsize=(10, 6))
    df['stay_group'] = pd.cut(df['length_of_stay'], 
                             bins=[0, 7, 30, 90, 180, float('inf')], 
                             labels=['1 week', '1 month', '3 months', '6 months', '6+ months'])
    sns.barplot(x='stay_group', y='two_year_recid', data=df)
    plt.title('Recidivism Rate by Length of Stay')
    plt.xticks(rotation=45)
    plt.ylabel('Recidivism Rate')
    plt.tight_layout()
    plt.show()
    
    print("\nRecidivism rates by length of stay:")
    print(df.groupby('stay_group')['two_year_recid'].agg(['count', 'mean']))

def print_correlation_matrix():
    numerical_vars = ['age', 'length_of_stay', 'two_year_recid']
    correlation_matrix = df[numerical_vars].corr()
    print("\nCorrelation matrix:")
    print(correlation_matrix)

# AIF360 Analysis
def fairness_analysis_on_race():
    # Create a copy of the dataframe and handle NA values
    df_fair = df.copy()
    df_fair = df_fair.dropna()
    
    # Convert categorical variables to numerical using one-hot encoding
    # Keep only necessary columns
    df_fair = df_fair[['race', 'two_year_recid']]
    
    # Create dummy variables for race
    race_dummies = pd.get_dummies(df_fair['race'])
    
    # Create the final dataframe with only necessary columns
    df_fair = pd.DataFrame({
        'two_year_recid': df_fair['two_year_recid'],
        'race': (df_fair['race'] == 'Caucasian').astype(int)  # 1 for Caucasian, 0 for others
    })
    
    protected_attribute = 'race'
    
    # Create AIF360 dataset with numerical data
    aif_dataset = BinaryLabelDataset(
        df=df_fair,
        label_names=['two_year_recid'],
        protected_attribute_names=[protected_attribute],
        favorable_label=0,
        unfavorable_label=1
    )
    
    # Calculate fairness metrics
    metrics = BinaryLabelDatasetMetric(
        aif_dataset, 
        unprivileged_groups=[{protected_attribute: 0}],  # non-Caucasian
        privileged_groups=[{protected_attribute: 1}]      # Caucasian
    )

    print("\n=== Fairness Metrics ===")
    print(f"Disparate Impact: {metrics.disparate_impact()}")
    print(f"Statistical Parity Difference: {metrics.statistical_parity_difference()}")

    # Apply the Reweighing algorithm to mitigate bias
    RW = Reweighing(
        unprivileged_groups=[{protected_attribute: 0}],
        privileged_groups=[{protected_attribute: 1}]
    )
    dataset_transformed = RW.fit_transform(aif_dataset)

    # Calculate metrics after transformation
    metrics_transformed = BinaryLabelDatasetMetric(
        dataset_transformed,
        unprivileged_groups=[{protected_attribute: 0}],
        privileged_groups=[{protected_attribute: 1}]
    )

    print("\n=== Fairness Metrics After Reweighing ===")
    print(f"Disparate Impact: {metrics_transformed.disparate_impact()}")
    print(f"Statistical Parity Difference: {metrics_transformed.statistical_parity_difference()}")

def perform_fairness_analysis():
    print("\n=== RACE-BASED FAIRNESS ANALYSIS ===")
    # Create a copy of the dataframe and handle NA values
    df_fair = df.copy()
    df_fair = df_fair.dropna()
    
    # Race analysis (existing code)
    df_race = pd.DataFrame({
        'two_year_recid': df_fair['two_year_recid'],
        'race': (df_fair['race'] == 'Caucasian').astype(int)
    })
    
    aif_dataset_race = BinaryLabelDataset(
        df=df_race,
        label_names=['two_year_recid'],
        protected_attribute_names=['race'],
        favorable_label=0,
        unfavorable_label=1
    )
    
    metrics_race = BinaryLabelDatasetMetric(
        aif_dataset_race, 
        unprivileged_groups=[{'race': 0}],
        privileged_groups=[{'race': 1}]
    )

    print("\nRace-based Metrics:")
    print(f"Disparate Impact: {metrics_race.disparate_impact()}")
    print(f"Statistical Parity Difference: {metrics_race.statistical_parity_difference()}")

    print("\n=== GENDER-BASED FAIRNESS ANALYSIS ===")
    # Gender analysis
    df_gender = pd.DataFrame({
        'two_year_recid': df_fair['two_year_recid'],
        'sex': (df_fair['sex'] == 'Female').astype(int)  # 1 for Female, 0 for Male
    })
    
    aif_dataset_gender = BinaryLabelDataset(
        df=df_gender,
        label_names=['two_year_recid'],
        protected_attribute_names=['sex'],
        favorable_label=0,
        unfavorable_label=1
    )
    
    metrics_gender = BinaryLabelDatasetMetric(
        aif_dataset_gender,
        unprivileged_groups=[{'sex': 0}],  # Male
        privileged_groups=[{'sex': 1}]      # Female
    )

    print("\nGender-based Metrics:")
    print(f"Disparate Impact: {metrics_gender.disparate_impact()}")
    print(f"Statistical Parity Difference: {metrics_gender.statistical_parity_difference()}")

    print("\n=== AGE-BASED FAIRNESS ANALYSIS ===")
    # Age analysis
    df_age = df_fair.copy()
    # Define young as below median age
    median_age = df_age['age'].median()
    df_age = pd.DataFrame({
        'two_year_recid': df_age['two_year_recid'],
        'age_group': (df_age['age'] >= median_age).astype(int)  # 1 for older, 0 for younger
    })
    
    aif_dataset_age = BinaryLabelDataset(
        df=df_age,
        label_names=['two_year_recid'],
        protected_attribute_names=['age_group'],
        favorable_label=0,
        unfavorable_label=1
    )
    
    metrics_age = BinaryLabelDatasetMetric(
        aif_dataset_age,
        unprivileged_groups=[{'age_group': 0}],  # younger
        privileged_groups=[{'age_group': 1}]      # older
    )

    print("\nAge-based Metrics:")
    print(f"Median age used as threshold: {median_age:.1f} years")
    print(f"Disparate Impact: {metrics_age.disparate_impact()}")
    print(f"Statistical Parity Difference: {metrics_age.statistical_parity_difference()}")

    # Optional: Print group sizes for context
    print("\nGroup Sizes:")
    print(f"Race - Caucasian: {df_race['race'].sum()}, Non-Caucasian: {len(df_race) - df_race['race'].sum()}")
    print(f"Gender - Female: {df_gender['sex'].sum()}, Male: {len(df_gender) - df_gender['sex'].sum()}")
    print(f"Age - Older: {df_age['age_group'].sum()}, Younger: {len(df_age) - df_age['age_group'].sum()}")

def main():
    print("=== COMPAS Dataset Analysis ===")
    
    # Basic data information
    print(f"\nDataset shape: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Visualizations
    plot_recidivism_by_race()
    plot_recidivism_by_age()
    plot_recidivism_by_sex()
    plot_recidivism_by_stay()
    print_correlation_matrix()
    
    # Fairness analysis
    # perform_fairness_analysis
    # fairness_analysis_on_race()
    perform_fairness_analysis()
    
if __name__ == "__main__":
    main() 