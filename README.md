# FINN Churn Analytics and Retention Insights

Welcome to FINN Churn Analytics—your end-to-end solution for understanding, predicting, and reducing customer churn in subscription-based vehicle services. Built with industry-standard tools and best practices, this toolkit lets you generate realistic customer data, develop a reliable churn model, and visualize actionable insights in a clean, interactive dashboard.

## Key Capabilities

* **Data Simulation**: Create synthetic customer records with detailed demographics, usage patterns, and subscription history.
* **Predictive Modeling**: Train a Gradient Boosting Classifier to estimate each customer’s churn risk, complete with performance metrics (AUC score, precision, recall) and a clear feature importance analysis.
* **Feature Engineering**: Enhance predictive power by deriving tenure, engagement flags, average spend, service frequency, and more.
* **Segmentation & Cohort Analysis**: Group customers by demographic or behavioral segments, track retention rates across acquisition cohorts, and compare lifetime value (LTV) trends month over month.
* **Retention Strategy Evaluation**: Identify high-risk customers, simulate targeted campaigns, and estimate return on investment for different interventions.
* **A/B Testing Framework**: Set up control and treatment groups to measure the impact of retention offers, compute effect sizes, and generate confidence intervals.
* **Interactive Dashboard**: Explore key metrics, charts, and tables through a self-contained HTML report (`finn_analytics_dashboard.html`) powered by Plotly and Chart.js.

## Prerequisites

* Python 3.8 or later
* `pip` for package management

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-org/finn-churn-analytics.git
   cd finn-churn-analytics
   ```
2. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

Run the main script to generate data, build the model, and produce the dashboard:

```bash
python finn_analytics.py
```

Then open `finn_analytics_dashboard.html` in a browser to review churn probabilities, segmentation insights, and retention metrics.

### Using the Analytics Class

You can also integrate the core functionality in your own projects:

```python
from finn_analytics import FINNChurnAnalytics

analytics = FINNChurnAnalytics()
analytics.generate_customer_data(n_customers=5000)
analytics.prepare_churn_features()
model_results = analytics.train_churn_model()
analytics.identify_at_risk_customers(threshold=0.7)
analytics.create_analytics_dashboard(output_path="dashboard.html")
```

## Repository Layout

```
├── finn_analytics.py               # Main module with data generation and modeling logic
├── finn_customer_data_complete.csv # Synthetic dataset
├── finn_analytics_dashboard.html   # Generated interactive report
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview and instructions
```

## Code Walkthrough

Below is a high-level explanation of the core code in `finn_analytics.py`. Each section shows key functions and how they fit together.

### 1. Data Simulation (`generate_customer_data`)

```python
def generate_customer_data(self, n_customers: int) -> pd.DataFrame:
    # 1. Create customer IDs and assign random demographics
    ids = range(1, n_customers + 1)
    ages = np.random.randint(18, 70, size=n_customers)
    regions = np.random.choice(['North', 'South', 'East', 'West'], n_customers)

    # 2. Simulate subscription start and end dates
    start_dates = np.random.choice(pd.date_range('2020-01-01', '2024-01-01'), n_customers)
    churn_flags = np.random.binomial(1, 0.2, size=n_customers)
    end_dates = [s + pd.DateOffset(months=np.random.randint(1, 24)) if f else pd.NaT
                 for s, f in zip(start_dates, churn_flags)]

    # 3. Assemble into a DataFrame
    data = pd.DataFrame({
        'customer_id': ids,
        'age': ages,
        'region': regions,
        'start_date': start_dates,
        'end_date': end_dates,
        'churned': churn_flags
    })
    return data
```

This function uses NumPy and Pandas to generate realistic customer profiles: demographics, randomly chosen start dates, and optional churn end dates when `churned == 1`.

### 2. Feature Engineering & Preparation (`prepare_churn_features`)

```python
def prepare_churn_features(self) -> pd.DataFrame:
    df = self.data.copy()
    # Calculate tenure in months
    df['tenure_months'] = ((df['end_date'].fillna(pd.Timestamp.today()) - df['start_date'])
                             / np.timedelta64(1, 'M')).astype(int)

    # Flag high engagement (e.g., >12 services per year)
    df['high_engagement'] = (df['services_per_year'] > 12).astype(int)

    # Merge or compute any additional features here
    features = df[['age', 'region_code', 'tenure_months', 'high_engagement']]
    labels = df['churned']
    return features, labels
```

This step transforms raw dates into numeric tenure, creates binary flags for engagement, and selects the final feature matrix for modeling.

### 3. Model Training (`train_churn_model`)

```python
def train_churn_model(self) -> dict:
    X, y = self.features, self.labels
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'auc': roc_auc_score(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred),
        'feature_importances': dict(zip(X.columns, model.feature_importances_))
    }
    self.model = model
    return metrics
```

This snippet splits data into train/test sets, fits a gradient boosting model, and computes AUC, detailed precision/recall, and feature importances to explain what drives churn predictions.

### 4. Identifying At-Risk Customers (`identify_at_risk_customers`)

```python
def identify_at_risk_customers(self, threshold: float = 0.7) -> pd.DataFrame:
    proba = self.model.predict_proba(self.features)[:, 1]
    self.data['churn_probability'] = proba
    at_risk = self.data[self.data['churn_probability'] >= threshold]
    return at_risk.sort_values('churn_probability', ascending=False)
```

By setting a probability threshold, this function flags customers most likely to churn, enabling targeted retention efforts.

### 5. Dashboard Creation (`create_analytics_dashboard`)

```python
def create_analytics_dashboard(self, output_path: str = 'finn_analytics_dashboard.html') -> None:
    # Build Plotly figures
    fig1 = px.histogram(self.data, x='churn_probability', nbins=20)
    fig2 = px.line(self.cohort_retention, x='month', y='retention_rate')

    # Render HTML with embedded charts
    with open(output_path, 'w') as f:
        f.write(self._render_template(fig1, fig2))
```

This method uses Plotly to generate interactive visualizations and writes a standalone HTML report combining charts, tables, and narrative insights.

---

## Contributing

We welcome improvements! To propose changes:

We welcome improvements! To propose changes:

1. Fork this repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Make your enhancements and add appropriate tests.
4. Submit a Pull Request with a clear description of your changes.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
