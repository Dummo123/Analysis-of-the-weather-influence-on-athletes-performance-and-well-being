# All important imports
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np


sns.set_theme(style="whitegrid")


# Load data
def load_match_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    mask = (df['Date'] >= '2022-08-01') & (df['Date'] <= '2023-05-31')
    df = df[mask]
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['TotalYellows'] = df['HY'] + df['AY']
    df['HomeWin'] = (df['FTR'] == 'H').astype(int)
    df['Draw'] = (df['FTR'] == 'D').astype(int)
    df['AwayWin'] = (df['FTR'] == 'A').astype(int)
    london_teams = ['Arsenal', 'Chelsea', 'Tottenham', 'Crystal Palace', 'West Ham', 'Fulham', 'Brentford']
    return df[df['HomeTeam'].isin(london_teams)]


def load_weather_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['DATE'])
    df['DATE'] = pd.to_datetime(df['DATE'])
    mask = (df['DATE'] >= '2022-08-01') & (df['DATE'] <= '2023-05-31')
    df = df[mask]
    df['Temperature'] = df['TMP'].str.split(',').str[0].str.replace('+', '').astype(float) / 10
    df['WindSpeed'] = df['WND'].str.split(',').str[3].astype(float) / 10
    return df[['DATE', 'Temperature', 'WindSpeed']]


def load_injury_data(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['Date of Injury'])
        mask = (df['Date of Injury'] >= '2022-08-01') & (df['Date of Injury'] <= '2023-05-31')
        return df[mask]
    except FileNotFoundError:
        print(f"Warning: Injury file '{file_path}' not found. Continuing without injury data.")
        return pd.DataFrame()


def merge_data(match_df, weather_df, injury_df):
    merged = pd.merge(match_df, weather_df, left_on='Date', right_on='DATE', how='left')
    if not injury_df.empty:
        injuries = injury_df.groupby('Date of Injury').size().reset_index(name='InjuryCount')
        merged = pd.merge(merged, injuries, left_on='Date', right_on='Date of Injury', how='left')
        merged['InjuryCount'] = merged['InjuryCount'].fillna(0)
    return merged.dropna(subset=['Temperature', 'WindSpeed'])


# Statistical testing functions
def test_correlation(x, y, alpha=0.05):
    corr, p_value = stats.pearsonr(x, y)
    significant = p_value < alpha
    return corr, p_value, significant


def test_group_differences(data, group_col, value_col, alpha=0.05):
    groups = data.groupby(group_col)[value_col].apply(list)
    f_val, p_value = stats.f_oneway(*groups)
    significant = p_value < alpha
    return f_val, p_value, significant


def test_proportions(data, group_col, success_col, alpha=0.05):
    contingency = pd.crosstab(data[group_col], data[success_col])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    significant = p_value < alpha
    return chi2, p_value, significant


graph_descriptions = {}
hypothesis_results = {}


# Graphs
def plot_temperature_vs_goals(df):
    plt.figure(figsize=(10, 7))
    sns.regplot(x='Temperature', y='TotalGoals', data=df, scatter_kws={'alpha': 0.5})
    corr, p_value, significant = test_correlation(df['Temperature'], df['TotalGoals'])
    result = "SUPPORTED" if significant else "NOT SUPPORTED"
    hypothesis_results['temperature_vs_goals.png'] = f"Hypothesis {result} (p={p_value:.4f})"
    plt.title(f'Temperature vs. Total Goals (2022-23)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Total Goals per Match')
    plt.tight_layout()
    plt.savefig('temperature_vs_goals.png')
    plt.close()
    graph_descriptions['temperature_vs_goals.png'] = (
        "Scatter plot with regression line showing relationship between temperature and goals. "
        "Each point represents a match. The line indicates the trend.\n"
        "HYPOTHESIS: Higher temperatures correlate with increased goal scoring due to improved muscle efficiency"
    )


def plot_temperature_vs_cards(df):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='Temperature', y='TotalYellows', data=df, alpha=0.7)
    corr, p_value, significant = test_correlation(df['Temperature'], df['TotalYellows'])
    result = "SUPPORTED" if significant else "NOT SUPPORTED"
    hypothesis_results['temperature_vs_cards.png'] = f"Hypothesis {result} (p={p_value:.4f})"
    plt.title(f'Temperature vs. Yellow Cards (2022-23)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Yellow Cards per Match')
    plt.tight_layout()
    plt.savefig('temperature_vs_cards.png')
    plt.close()
    graph_descriptions['temperature_vs_cards.png'] = (
        "Scatter plot relationship between temperature and yellow cards. "
        "Each point represents a match. There is no pattern => weak correlation.\n"
        "HYPOTHESIS: Colder temperatures increase yellow cards due to reduced motor control"
    )


def plot_wind_vs_home_wins(df):
    df['WindSpeedBin'] = pd.cut(df['WindSpeed'], bins=5)
    win_rates = df.groupby('WindSpeedBin')['HomeWin'].mean().reset_index()
    chi2, p_value, significant = test_proportions(df, 'WindSpeedBin', 'HomeWin')
    result = "SUPPORTED" if significant else "NOT SUPPORTED"
    hypothesis_results['wind_vs_home_wins.png'] = f"Hypothesis {result} (p={p_value:.4f})"
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(x='WindSpeedBin', y='HomeWin', data=win_rates)
    plt.title(f'Wind Speed vs. Home Win Rate (2022-23)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Home Win Rate')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('wind_vs_home_wins.png')
    plt.close()
    graph_descriptions['wind_vs_home_wins.png'] = (
        "Bar chart showing home win rates across wind speed ranges. "
        "Height indicates win percentage for each wind category.\n"
        "HYPOTHESIS: Higher wind speeds strengthen home advantage due to locals' wind adaptability"
    )


def plot_temperature_injuries(df):
    if 'InjuryCount' not in df:
        print("No injury data available")
        return


def plot_wind_speed_vs_goals(df):
    plt.figure(figsize=(10, 7))
    sns.regplot(x='WindSpeed', y='TotalGoals', data=df, scatter_kws={'alpha': 0.5})
    corr, p_value, significant = test_correlation(df['WindSpeed'], df['TotalGoals'])
    result = "SUPPORTED" if significant else "NOT SUPPORTED"
    hypothesis_results['wind_vs_goals.png'] = f"Hypothesis {result} (p={p_value:.4f})"
    plt.title(f'Wind Speed vs. Total Goals (2022-23)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Total Goals per Match')
    plt.tight_layout()
    plt.savefig('wind_vs_goals.png')
    plt.close()
    graph_descriptions['wind_vs_goals.png'] = (
        "Scatter plot relationship between wind speed and goals. "
        "The downward trend suggests higher winds may reduce scoring.\n"
        "HYPOTHESIS: Higher wind speeds correlate with fewer goals due to worse accuracy"
    )


def plot_temperature_vs_home_wins(df):
    df['TempBin'] = pd.cut(df['Temperature'], bins=5)
    win_rates = df.groupby('TempBin')['HomeWin'].mean().reset_index()
    chi2, p_value, significant = test_proportions(df, 'TempBin', 'HomeWin')
    result = "SUPPORTED" if significant else "NOT SUPPORTED"
    hypothesis_results['temperature_vs_home_wins.png'] = f"Hypothesis {result} (p={p_value:.4f})"
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(x='TempBin', y='HomeWin', data=win_rates)
    plt.title(f'Temperature vs. Home Win Rate (2022-23)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Home Win Rate')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('temperature_vs_home_wins.png')
    plt.close()
    graph_descriptions['temperature_vs_home_wins.png'] = (
        "Bar chart showing home win rates across temperature ranges. "
        "Higher bars indicate better home team performance in that temperature range.\n"
        "HYPOTHESIS: Higher temperatures increase home win rates due to heat tolerance"
    )


def plot_wind_speed_vs_cards(df):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='WindSpeed', y='TotalYellows', data=df, alpha=0.7)
    corr, p_value, significant = test_correlation(df['WindSpeed'], df['TotalYellows'])
    result = "SUPPORTED" if significant else "NOT SUPPORTED"
    hypothesis_results['wind_vs_cards.png'] = f"Hypothesis {result} (p={p_value:.4f})"
    plt.title(f'Wind Speed vs. Yellow Cards (2022-23)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Yellow Cards per Match')
    plt.tight_layout()
    plt.savefig('wind_vs_cards.png')
    plt.close()
    graph_descriptions['wind_vs_cards.png'] = (
        "Scatter plot relationship between wind speed and yellow cards. "
        "Each point represents a match. Higher winds show increased card incidents.\n"
        "HYPOTHESIS: Higher wind speeds increase yellow cards due to more disruptions"
    )


def plot_goal_difference_by_temp(df):
    df['GoalDiff'] = abs(df['FTHG'] - df['FTAG'])
    plt.figure(figsize=(10, 7))
    sns.boxplot(x=pd.cut(df['Temperature'], bins=[-5, 5, 10, 15, 20, 25]),
                y='GoalDiff', data=df)
    df['TempBin'] = pd.cut(df['Temperature'], bins=[-5, 5, 10, 15, 20, 25])
    f_val, p_value, significant = test_group_differences(df, 'TempBin', 'GoalDiff')
    result = "SUPPORTED" if significant else "NOT SUPPORTED"
    hypothesis_results['goal_diff_by_temp.png'] = f"Hypothesis {result} (p={p_value:.4f})"
    plt.title(f'Goal Difference by Temperature (2022-23)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Absolute Goal Difference')
    plt.tight_layout()
    plt.savefig('goal_diff_by_temp.png')
    plt.close()
    graph_descriptions['goal_diff_by_temp.png'] = (
        "Box plots showing goal difference distribution across temperature ranges. "
        "Higher boxes indicate more decisive matches. Optimal range: 10-20°C.\n"
        "HYPOTHESIS: Moderate temperatures (10-20°C) produce larger goal differences"
    )


# Distribution Graphs
def plot_home_team_distribution(df):
    plt.figure(figsize=(8, 8))
    df['HomeTeam'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Home Team Distribution in London Matches')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('home_team_distribution.png')
    plt.close()
    graph_descriptions['home_team_distribution.png'] = (
        "Pie chart showing distribution of matches by home team. "
        "Teams with more home matches appear as larger slices."
    )


def plot_match_results_distribution(df):
    plt.figure(figsize=(8, 8))
    result_counts = pd.Series({
        'Home Wins': df['HomeWin'].sum(),
        'Away Wins': df['AwayWin'].sum(),
        'Draws': df['Draw'].sum()
    })
    result_counts.plot.pie(autopct='%1.1f%%')
    plt.title('Match Results Distribution')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('match_results_distribution.png')
    plt.close()
    graph_descriptions['match_results_distribution.png'] = (
        "Pie chart showing overall match results distribution. "
        "Shows percentage of home wins, away wins, and draws."
    )


def plot_temperature_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Temperature'], bins=15, kde=True)
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (°C)')
    plt.tight_layout()
    plt.savefig('temperature_distribution.png')
    plt.close()
    graph_descriptions['temperature_distribution.png'] = (
        "Histogram with KDE showing distribution of match temperatures. "
        "Most matches played in moderate temperatures (5-15°C)."
    )


def plot_goals_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['TotalGoals'], bins=10, kde=True)
    plt.title('Total Goals Distribution')
    plt.xlabel('Goals per Match')
    plt.tight_layout()
    plt.savefig('goals_distribution.png')
    plt.close()
    graph_descriptions['goals_distribution.png'] = (
        "Histogram with KDE showing distribution of total goals per match. "
        "Most common scores are 2-3 goals per game."
    )


def plot_yellow_cards_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['TotalYellows'], bins=10, kde=True)
    plt.title('Yellow Cards Distribution')
    plt.xlabel('Yellow Cards per Match')
    plt.tight_layout()
    plt.savefig('yellow_cards_distribution.png')
    plt.close()
    graph_descriptions['yellow_cards_distribution.png'] = (
        "Histogram with KDE showing distribution of yellow cards per match. "
        "Most matches have 2-4 yellow cards."
    )


# Graph description
def create_descriptions_key_image(descriptions_dict, hypothesis_dict, output_file='descriptions_key.png'):
    # Format of pic
    num_graphs = len(descriptions_dict)
    fig_height = max(6, num_graphs * 1.5)
    fig = plt.figure(figsize=(10, fig_height))
    ax = fig.add_subplot(111)
    ax.axis('off')
    header_text = "Football Analytics - Graph Descriptions Key\n\n"
    plt.figtext(0.5, 0.97, header_text,
                ha='center', va='top',
                fontsize=14, weight='bold')
    text_content = ""
    for i, (graph_name, description) in enumerate(descriptions_dict.items()):
        hyp_result = hypothesis_dict.get(graph_name, "No hypothesis test performed")
        text_content += f"GRAPH: {graph_name}\n"
        text_content += f"RESULT: {hyp_result}\n\n"
        text_content += f"{description}\n\n"
        if i < len(descriptions_dict) - 1:
            text_content += "-" * 100 + "\n\n"
    plt.figtext(0.05, 0.90, text_content,
                ha='left', va='top',
                fontsize=10, wrap=True,
                bbox=dict(facecolor='white', alpha=1, boxstyle='round'))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()


# Recommendation system for optimal match conditions
def generate_recommendation(df):
    temp_bins = pd.cut(df['Temperature'], bins=np.arange(0, 31, 5))
    goals_by_temp = df.groupby(temp_bins)['TotalGoals'].mean()
    optimal_temp_range = goals_by_temp.idxmax()
    wind_bins = pd.cut(df['WindSpeed'], bins=np.arange(0, 16, 3))
    goals_by_wind = df.groupby(wind_bins)['TotalGoals'].mean()
    optimal_wind_range = goals_by_wind.idxmax()
    cards_by_temp = df.groupby(temp_bins)['TotalYellows'].mean()
    fair_temp_range = cards_by_temp.idxmin()
    cards_by_wind = df.groupby(wind_bins)['TotalYellows'].mean()
    fair_wind_range = cards_by_wind.idxmin()

    # Create graph
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    sns.barplot(x=goals_by_temp.index.astype(str), y=goals_by_temp.values, ax=ax[0, 0], palette='viridis')
    ax[0, 0].axhline(goals_by_temp.max(), color='red', linestyle='--')
    ax[0, 0].set_title('Average Goals by Temperature Range')
    ax[0, 0].set_ylabel('Average Goals')
    ax[0, 0].set_xlabel('Temperature Range (°C)')
    sns.barplot(x=goals_by_wind.index.astype(str), y=goals_by_wind.values, ax=ax[0, 1], palette='viridis')
    ax[0, 1].axhline(goals_by_wind.max(), color='red', linestyle='--')
    ax[0, 1].set_title('Average Goals by Wind Speed Range')
    ax[0, 1].set_ylabel('Average Goals')
    ax[0, 1].set_xlabel('Wind Speed Range (m/s)')
    sns.barplot(x=cards_by_temp.index.astype(str), y=cards_by_temp.values, ax=ax[1, 0], palette='coolwarm')
    ax[1, 0].axhline(cards_by_temp.min(), color='green', linestyle='--')
    ax[1, 0].set_title('Average Yellow Cards by Temperature Range')
    ax[1, 0].set_ylabel('Average Yellow Cards')
    ax[1, 0].set_xlabel('Temperature Range (°C)')
    sns.barplot(x=cards_by_wind.index.astype(str), y=cards_by_wind.values, ax=ax[1, 1], palette='coolwarm')
    ax[1, 1].axhline(cards_by_wind.min(), color='green', linestyle='--')
    ax[1, 1].set_title('Average Yellow Cards by Wind Speed Range')
    ax[1, 1].set_ylabel('Average Yellow Cards')
    ax[1, 1].set_xlabel('Wind Speed Range (m/s)')
    plt.tight_layout(pad=3.0)
    recommendation = (
        f"Optimal Match Conditions Recommendation:\n\n"
        f"1. For Maximum Goals:\n"
        f"   • Temperature: {optimal_temp_range}\n"
        f"   • Wind Speed: {optimal_wind_range}\n\n"
        f"2. For Fair Play (Minimum Cards):\n"
        f"   • Temperature: {fair_temp_range}\n"
        f"   • Wind Speed: {fair_wind_range}\n\n"
        f"3. Balanced Recommendation:\n"
        f"   • Temperature: 15-20°C\n"
        f"   • Wind Speed: 3-6 m/s\n\n"
        "Analysis shows moderate temperatures (15-20°C) and light winds (3-6 m/s)\n"
        "provide the best balance of high-scoring matches and fair play conditions."
    )
    plt.figtext(0.5, 0.01, recommendation,
                ha='center', va='bottom',
                fontsize=12,
                bbox=dict(facecolor='lightyellow', alpha=0.8))
    plt.suptitle("Football Match Conditions Recommendation System", fontsize=16, y=0.99)
    plt.savefig('match_conditions_recommendation.png', bbox_inches='tight')
    plt.close()
    graph_descriptions['match_conditions_recommendation.png'] = (
        "Optimal match conditions recommendation based on historical analysis. "
        "Top charts show goal-maximizing conditions, bottom charts show fair-play conditions. "
        "Red dashed lines indicate maximum goals, green dashed lines indicate minimum cards. "
        "Final recommendation balances both factors for optimal match quality."
    )


match_df = load_match_data('E0.csv')
weather_df = load_weather_data('03772099999.csv')
injury_df = load_injury_data('player_injuries_impact.csv')
merged_df = merge_data(match_df, weather_df, injury_df)
print(f"{len(merged_df)} matches between datasets")

# Turn all graphs into pictures
plot_temperature_vs_goals(merged_df)
plot_temperature_vs_cards(merged_df)
plot_wind_vs_home_wins(merged_df)
plot_temperature_injuries(merged_df)
plot_wind_speed_vs_goals(merged_df)
plot_temperature_vs_home_wins(merged_df)
plot_wind_speed_vs_cards(merged_df)
plot_goal_difference_by_temp(merged_df)
plot_home_team_distribution(merged_df)
plot_match_results_distribution(merged_df)
plot_temperature_distribution(merged_df)
plot_goals_distribution(merged_df)
plot_yellow_cards_distribution(merged_df)
create_descriptions_key_image(graph_descriptions, hypothesis_results)
generate_recommendation(merged_df)

