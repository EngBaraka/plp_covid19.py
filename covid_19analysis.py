# %% [markdown]
# # COVID-19 Global Data Analysis
# **Author**: [BARAKA ELISHAMA OBIERO]  
# **Date**: [08/05/2025]

# %%
# Import libraries with error handling
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display, Markdown
    import plotly.express as px
    import requests
    from io import StringIO
    print("✅ Libraries imported successfully!")
except ImportError as e:
    print(f"❌ Missing library: {e}\nPlease install with: pip install pandas matplotlib seaborn plotly ipython")

# %%
# %% [markdown]
# ## Data Loading with Robust Error Handling

# %%
def load_covid_data():
    """Load COVID-19 data with multiple fallback options"""
    sources = [
        "https://covid.ourworldindata.org/data/owid-covid-data.csv",
        "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
        "local_backup/owid-covid-data.csv"  # Local fallback
    ]
    
    for url in sources:
        try:
            if url.startswith('http'):
                print(f"Attempting to download from {url}")
                response = requests.get(url)
                response.raise_for_status()
                return pd.read_csv(StringIO(response.text))
            else:
                print(f"Loading from local file: {url}")
                return pd.read_csv(url)
        except Exception as e:
            print(f"⚠️ Failed to load from {url}: {str(e)}")
            continue
    
    raise Exception("All data sources failed. Please check your internet connection.")

try:
    covid_df = load_covid_data()
    print(f"\n✅ Data loaded successfully! Shape: {covid_df.shape}")
    
    # Save a local backup
    covid_df.to_csv("local_backup/owid-covid-data.csv", index=False)
    print("📁 Local backup saved")
    
except Exception as e:
    print(f"\n❌ Critical error loading data: {str(e)}")
    raise

# %%
# %% [markdown]
# ## Data Cleaning and Preparation

# %%
# Convert date column
covid_df['date'] = pd.to_datetime(covid_df['date'])

# Select countries of interest
countries_of_interest = [
    'Kenya', 'United States', 'India', 
    'Brazil', 'United Kingdom', 'Germany', 
    'South Africa'
]

# Filter and clean data
filtered_df = (
    covid_df
    .query("location in @countries_of_interest")
    .copy()
    .sort_values(['location', 'date'])
)

# Forward fill missing values for key metrics
metrics_to_clean = [
    'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
    'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
    'population'
]

for metric in metrics_to_clean:
    filtered_df[metric] = filtered_df.groupby('location')[metric].ffill().fillna(0)

# Calculate derived metrics
filtered_df['death_rate'] = (
    filtered_df['total_deaths'] / filtered_df['total_cases'].replace(0, 1)
)
filtered_df['vaccination_percentage'] = (
    filtered_df['people_vaccinated'] / filtered_df['population'] * 100
)
filtered_df['full_vaccination_percentage'] = (
    filtered_df['people_fully_vaccinated'] / filtered_df['population'] * 100
)

# Get latest records
latest_data = filtered_df.groupby('location').last().reset_index()

print("\n🧹 Data cleaning complete!")
print(f"📅 Date range: {filtered_df['date'].min().date()} to {filtered_df['date'].max().date()}")
print(f"🌍 Countries analyzed: {', '.join(countries_of_interest)}")

# %%
# %% [markdown]
# ## Visualization: Case Trends

# %%
# Configure visualization settings
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 7)

# %%
# Total cases over time
fig1, ax1 = plt.subplots()
sns.lineplot(
    data=filtered_df,
    x='date',
    y='total_cases',
    hue='location',
    ax=ax1,
    linewidth=2.5
)
ax1.set(
    title='Total COVID-19 Cases Over Time',
    xlabel='Date',
    ylabel='Total Cases (log scale)',
    yscale='log'
)
ax1.legend(title='Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/total_cases.png', dpi=300)
plt.show()

# %%
# New cases (7-day average)
fig2, ax2 = plt.subplots()
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    ax2.plot(
        country_data['date'],
        country_data['new_cases'].rolling(7).mean(),
        label=country,
        linewidth=2
    )
    
ax2.set(
    title='Daily New Cases (7-Day Moving Average)',
    xlabel='Date',
    ylabel='New Cases'
)
ax2.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/new_cases.png', dpi=300)
plt.show()

# %%
# %% [markdown]
# ## Visualization: Vaccination Progress

# %%
# Vaccination progress
fig3, ax3 = plt.subplots()
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    ax3.plot(
        country_data['date'],
        country_data['full_vaccination_percentage'],
        label=country,
        linewidth=2.5
    )
    
ax3.set(
    title='Full Vaccination Progress',
    xlabel='Date',
    ylabel='% Population Fully Vaccinated'
)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/vaccination_progress.png', dpi=300)
plt.show()

# %%
# %% [markdown]
# ## Interactive Visualization

# %%
# Interactive choropleth map
try:
    fig = px.choropleth(
        latest_data,
        locations="iso_code",
        color="full_vaccination_percentage",
        hover_name="location",
        hover_data=["total_cases", "total_deaths", "population"],
        color_continuous_scale=px.colors.diverging.RdYlGn,
        title="Global Vaccination Progress",
        labels={'full_vaccination_percentage': '% Fully Vaccinated'}
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    fig.show()
    fig.write_html("visualizations/vaccination_map.html")
except Exception as e:
    print(f"⚠️ Could not create interactive map: {str(e)}")

# %%
# %% [markdown]
# ## Key Insights

# %%
insights = f"""
### COVID-19 Analysis Insights (as of {latest_data['date'].max().date()})

1. **Case Trends**:
   - Highest total cases: {latest_data.loc[latest_data['total_cases'].idxmax(), 'location']} ({latest_data['total_cases'].max():,})
   - Lowest vaccination rate: {latest_data.loc[latest_data['full_vaccination_percentage'].idxmin(), 'location']} ({latest_data['full_vaccination_percentage'].min():.1f}%)

2. **Vaccination Progress**:
   - Most vaccinated: {latest_data.loc[latest_data['full_vaccination_percentage'].idxmax(), 'location']} ({latest_data['full_vaccination_percentage'].max():.1f}%)
   - Average vaccination rate: {latest_data['full_vaccination_percentage'].mean():.1f}%

3. **Mortality**:
   - Highest death rate: {latest_data.loc[latest_data['death_rate'].idxmax(), 'location']} ({latest_data['death_rate'].max():.2%})
   - Lowest death rate: {latest_data.loc[latest_data['death_rate'].idxmin(), 'location']} ({latest_data['death_rate'].min():.2%})

4. **Regional Observations**:
   - African nations showed delayed but steep case curves
   - European nations achieved faster vaccination rollout
"""

display(Markdown(insights))

# Save insights to file
with open('output/insights.md', 'w') as f:
    f.write(insights)

print("\n📊 Analysis complete! Check the visualizations/ folder for output files.")
