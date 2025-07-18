"""
FINN Customer Churn Prediction and Retention Analytics
=====================================================

This project demonstrates advanced customer analytics skills relevant to FINN's subscription model:
- Customer churn prediction using machine learning
- Customer segmentation and lifetime value analysis
- Retention campaign optimization
- Cohort analysis for subscription business
- A/B testing framework for retention strategies

Tech Stack: Python, Pandas, Scikit-learn, Plotly, Advanced Analytics
"""

"""
COMPLETE FINN Customer Churn Prediction and Retention Analytics
==============================================================

This is the FULL, COMPLETE, WORKING version that generates HTML files.
Just copy this entire code and run it - no modifications needed!

Tech Stack: Python, Pandas, Scikit-learn, Plotly, Advanced Analytics
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

class FINNChurnAnalytics:
    def __init__(self):
        self.data = None
        self.churn_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def generate_customer_data(self, n_customers=10000):
        """Generate synthetic FINN customer data"""
        np.random.seed(42)
        
        # Customer demographics
        customer_ids = [f"FINN_{i:06d}" for i in range(n_customers)]
        acquisition_dates = pd.date_range(start='2022-01-01', end='2024-12-31', periods=n_customers)
        
        # Customer segments
        customer_segments = np.random.choice(['Urban_Professional', 'Young_Family', 'Corporate_User', 'Eco_Conscious', 'Premium_User'], 
                                           n_customers, p=[0.35, 0.25, 0.20, 0.15, 0.05])
        
        # Geographic distribution
        cities = np.random.choice(['Munich', 'Berlin', 'Hamburg', 'Frankfurt', 'Stuttgart', 'Cologne'], 
                                n_customers, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
        
        # Vehicle preferences
        vehicle_types = np.random.choice(['Compact_Electric', 'Mid_Size_ICE', 'SUV_Electric', 'Luxury_ICE', 'Commercial_Van'], 
                                       n_customers, p=[0.30, 0.25, 0.20, 0.15, 0.10])
        
        customers = []
        
        for i in range(n_customers):
            # Base characteristics
            segment = customer_segments[i]
            city = cities[i]
            vehicle_type = vehicle_types[i]
            acq_date = acquisition_dates[i]
            
            # Segment-specific behaviors
            if segment == 'Urban_Professional':
                avg_monthly_price = np.random.normal(650, 100)
                subscription_length = np.random.normal(12, 4)
                usage_intensity = np.random.uniform(0.6, 0.9)
                price_sensitivity = np.random.uniform(0.3, 0.7)
            elif segment == 'Young_Family':
                avg_monthly_price = np.random.normal(750, 150)
                subscription_length = np.random.normal(18, 6)
                usage_intensity = np.random.uniform(0.7, 1.0)
                price_sensitivity = np.random.uniform(0.5, 0.8)
            elif segment == 'Corporate_User':
                avg_monthly_price = np.random.normal(800, 120)
                subscription_length = np.random.normal(24, 8)
                usage_intensity = np.random.uniform(0.8, 1.0)
                price_sensitivity = np.random.uniform(0.1, 0.4)
            elif segment == 'Eco_Conscious':
                avg_monthly_price = np.random.normal(600, 80)
                subscription_length = np.random.normal(15, 5)
                usage_intensity = np.random.uniform(0.5, 0.8)
                price_sensitivity = np.random.uniform(0.4, 0.7)
            else:  # Premium_User
                avg_monthly_price = np.random.normal(1200, 200)
                subscription_length = np.random.normal(20, 6)
                usage_intensity = np.random.uniform(0.4, 0.7)
                price_sensitivity = np.random.uniform(0.1, 0.3)
            
            # Behavioral metrics
            total_subscriptions = max(1, int(np.random.poisson(2)))
            avg_subscription_length = max(1, subscription_length)
            support_tickets = np.random.poisson(1.5)
            app_engagement_score = np.random.uniform(0.2, 1.0)
            
            # Calculate tenure
            tenure_days = (datetime.now() - acq_date).days
            
            # Vehicle swaps (FINN's flexible model)
            vehicle_swaps = np.random.poisson(0.5 * tenure_days / 365)
            
            # Payment behavior
            late_payments = np.random.poisson(0.3)
            payment_method_changes = np.random.poisson(0.2)
            
            # Churn probability based on various factors
            churn_prob = 0.05  # Base churn rate
            
            # Adjust based on segment
            if segment == 'Corporate_User':
                churn_prob *= 0.6  # Lower churn
            elif segment == 'Young_Family':
                churn_prob *= 0.8
            elif segment == 'Premium_User':
                churn_prob *= 0.7
            
            # Adjust based on behaviors
            if support_tickets > 3:
                churn_prob *= 1.5
            if late_payments > 2:
                churn_prob *= 1.3
            if app_engagement_score < 0.3:
                churn_prob *= 1.4
            if avg_subscription_length < 6:
                churn_prob *= 1.2
            if vehicle_swaps > 2:
                churn_prob *= 0.8  # Engaged customers
            
            # Final churn decision
            churned = np.random.random() < min(churn_prob, 0.4)
            
            # Calculate CLV
            clv = avg_monthly_price * avg_subscription_length * total_subscriptions * (1 - churn_prob)
            
            customers.append({
                'customer_id': customer_ids[i],
                'acquisition_date': acq_date,
                'customer_segment': segment,
                'city': city,
                'preferred_vehicle_type': vehicle_type,
                'tenure_days': tenure_days,
                'total_subscriptions': total_subscriptions,
                'avg_subscription_length_months': avg_subscription_length,
                'avg_monthly_price_eur': avg_monthly_price,
                'vehicle_swaps': vehicle_swaps,
                'support_tickets': support_tickets,
                'late_payments': late_payments,
                'payment_method_changes': payment_method_changes,
                'app_engagement_score': app_engagement_score,
                'usage_intensity': usage_intensity,
                'price_sensitivity': price_sensitivity,
                'customer_lifetime_value': clv,
                'churned': churned
            })
        
        self.data = pd.DataFrame(customers)
        return self.data
    
    def prepare_churn_features(self):
        """Prepare features for churn prediction model"""
        # Create derived features
        self.data['avg_monthly_spend'] = self.data['avg_monthly_price_eur'] * self.data['usage_intensity']
        self.data['tenure_months'] = self.data['tenure_days'] / 30
        self.data['subscriptions_per_year'] = self.data['total_subscriptions'] / (self.data['tenure_months'] / 12)
        self.data['support_tickets_per_month'] = self.data['support_tickets'] / self.data['tenure_months']
        self.data['swaps_per_subscription'] = self.data['vehicle_swaps'] / self.data['total_subscriptions']
        
        # Engagement metrics
        self.data['high_engagement'] = (self.data['app_engagement_score'] > 0.7) & (self.data['vehicle_swaps'] > 0)
        self.data['payment_issues'] = (self.data['late_payments'] > 1) | (self.data['payment_method_changes'] > 1)
        
        # Segment-based features
        self.data['is_premium_segment'] = self.data['customer_segment'].isin(['Premium_User', 'Corporate_User'])
        self.data['is_price_sensitive'] = self.data['price_sensitivity'] > 0.6
        
        # Encode categorical variables
        categorical_cols = ['customer_segment', 'city', 'preferred_vehicle_type']
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[f'{col}_encoded'] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le
        
        return self.data
    
    def train_churn_model(self):
        """Train churn prediction model"""
        # Select features
        feature_cols = [
            'tenure_months', 'total_subscriptions', 'avg_subscription_length_months',
            'avg_monthly_spend', 'vehicle_swaps', 'support_tickets_per_month',
            'app_engagement_score', 'usage_intensity', 'price_sensitivity',
            'subscriptions_per_year', 'swaps_per_subscription',
            'high_engagement', 'payment_issues', 'is_premium_segment', 'is_price_sensitive',
            'customer_segment_encoded', 'city_encoded', 'preferred_vehicle_type_encoded'
        ]
        
        X = self.data[feature_cols]
        y = self.data['churned']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.churn_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
        self.churn_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.churn_model.predict(X_test_scaled)
        y_pred_proba = self.churn_model.predict_proba(X_test_scaled)[:, 1]
        
        # Model evaluation
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Churn Model Performance:")
        print(f"AUC Score: {auc_score:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, self.churn_model.feature_importances_))
        
        return {
            'auc_score': auc_score,
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }
    
    def customer_segmentation_analysis(self):
        """Analyze customer segments and their characteristics"""
        segment_analysis = self.data.groupby('customer_segment').agg({
            'customer_lifetime_value': 'mean',
            'avg_monthly_price_eur': 'mean',
            'avg_subscription_length_months': 'mean',
            'churned': 'mean',
            'app_engagement_score': 'mean',
            'support_tickets': 'mean',
            'vehicle_swaps': 'mean'
        }).round(2)
        
        return segment_analysis
    
    def cohort_analysis(self):
        """Perform cohort analysis for retention insights"""
        # Create cohort based on acquisition month
        self.data['acquisition_month'] = self.data['acquisition_date'].dt.to_period('M')
        
        # Calculate retention by cohort
        cohort_data = self.data.groupby('acquisition_month').agg({
            'customer_id': 'count',
            'churned': 'sum',
            'customer_lifetime_value': 'mean'
        }).reset_index()
        
        cohort_data['retention_rate'] = 1 - (cohort_data['churned'] / cohort_data['customer_id'])
        cohort_data['acquisition_month'] = cohort_data['acquisition_month'].astype(str)
        
        return cohort_data
    
    def identify_at_risk_customers(self, risk_threshold=0.7):
        """Identify customers at high risk of churning"""
        if self.churn_model is None:
            raise ValueError("Churn model not trained yet. Run train_churn_model() first.")
        
        # Get feature columns
        feature_cols = [
            'tenure_months', 'total_subscriptions', 'avg_subscription_length_months',
            'avg_monthly_spend', 'vehicle_swaps', 'support_tickets_per_month',
            'app_engagement_score', 'usage_intensity', 'price_sensitivity',
            'subscriptions_per_year', 'swaps_per_subscription',
            'high_engagement', 'payment_issues', 'is_premium_segment', 'is_price_sensitive',
            'customer_segment_encoded', 'city_encoded', 'preferred_vehicle_type_encoded'
        ]
        
        # Predict churn probability for all customers
        X = self.data[feature_cols]
        X_scaled = self.scaler.transform(X)
        churn_probabilities = self.churn_model.predict_proba(X_scaled)[:, 1]
        
        # Add to dataframe
        self.data['churn_probability'] = churn_probabilities
        
        # Identify at-risk customers
        at_risk_customers = self.data[self.data['churn_probability'] > risk_threshold].copy()
        at_risk_customers = at_risk_customers.sort_values('churn_probability', ascending=False)
        
        return at_risk_customers[['customer_id', 'customer_segment', 'city', 'customer_lifetime_value', 
                                 'churn_probability', 'avg_monthly_price_eur', 'support_tickets']]
    
    def create_analytics_dashboard(self):
        """Create comprehensive churn analytics dashboard"""
        # Get analyses
        segment_analysis = self.customer_segmentation_analysis()
        cohort_data = self.cohort_analysis()
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Churn Rate by Segment', 'Customer Lifetime Value by Segment',
                          'Cohort Retention Analysis', 'Churn Probability Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Churn rate by segment
        fig.add_trace(
            go.Bar(x=segment_analysis.index, y=segment_analysis['churned'],
                   name='Churn Rate', marker_color='red', opacity=0.7),
            row=1, col=1
        )
        
        # CLV by segment
        fig.add_trace(
            go.Bar(x=segment_analysis.index, y=segment_analysis['customer_lifetime_value'],
                   name='CLV', marker_color='green', opacity=0.7),
            row=1, col=2
        )
        
        # Cohort retention over time
        fig.add_trace(
            go.Scatter(x=cohort_data['acquisition_month'], y=cohort_data['retention_rate'],
                      mode='lines+markers', name='Retention Rate', line=dict(color='blue')),
            row=2, col=1
        )
        
        # Churn probability distribution
        if 'churn_probability' in self.data.columns:
            fig.add_trace(
                go.Histogram(x=self.data['churn_probability'], name='Churn Probability',
                            marker_color='orange', opacity=0.7, nbinsx=30),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="FINN Customer Analytics Dashboard",
            showlegend=False,
            height=800
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Churn Rate", row=1, col=1)
        fig.update_yaxes(title_text="CLV (€)", row=1, col=2)
        fig.update_yaxes(title_text="Retention Rate", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        # SAVE HTML FILE - THIS IS THE KEY LINE!
        fig.write_html("finn_analytics_dashboard.html")
        print("✅ HTML Dashboard saved as: finn_analytics_dashboard.html")
        
        return fig
    
    def retention_campaign_optimizer(self):
        """Optimize retention campaigns based on customer segments and churn risk"""
        if 'churn_probability' not in self.data.columns:
            raise ValueError("Run identify_at_risk_customers() first to calculate churn probabilities.")
        
        # Define intervention strategies
        interventions = {
            'High_Risk_Premium': {
                'criteria': lambda df: (df['churn_probability'] > 0.7) & (df['is_premium_segment']),
                'strategy': 'Personal Account Manager + Exclusive Perks',
                'cost_per_customer': 150,
                'expected_retention_lift': 0.25
            },
            'High_Risk_Price_Sensitive': {
                'criteria': lambda df: (df['churn_probability'] > 0.6) & (df['is_price_sensitive']),
                'strategy': 'Discount Offer + Flexible Terms',
                'cost_per_customer': 80,
                'expected_retention_lift': 0.20
            },
            'Medium_Risk_Low_Engagement': {
                'criteria': lambda df: (df['churn_probability'] > 0.4) & (df['app_engagement_score'] < 0.4),
                'strategy': 'App Engagement Campaign + Tutorial',
                'cost_per_customer': 25,
                'expected_retention_lift': 0.15
            },
            'Support_Heavy_Users': {
                'criteria': lambda df: (df['support_tickets'] > 3) & (df['churn_probability'] > 0.5),
                'strategy': 'Proactive Support + Service Credit',
                'cost_per_customer': 60,
                'expected_retention_lift': 0.18
            }
        }
        
        campaign_results = []
        
        for campaign_name, config in interventions.items():
            # Identify target customers
            target_customers = self.data[config['criteria'](self.data)]
            
            if len(target_customers) > 0:
                # Calculate campaign metrics
                total_clv_at_risk = target_customers['customer_lifetime_value'].sum()
                expected_clv_saved = total_clv_at_risk * config['expected_retention_lift']
                total_campaign_cost = len(target_customers) * config['cost_per_customer']
                roi = (expected_clv_saved - total_campaign_cost) / total_campaign_cost if total_campaign_cost > 0 else 0
                
                campaign_results.append({
                    'campaign': campaign_name,
                    'strategy': config['strategy'],
                    'target_customers': len(target_customers),
                    'total_clv_at_risk': total_clv_at_risk,
                    'expected_clv_saved': expected_clv_saved,
                    'campaign_cost': total_campaign_cost,
                    'roi': roi,
                    'cost_per_customer': config['cost_per_customer']
                })
        
        return pd.DataFrame(campaign_results).sort_values('roi', ascending=False)
    
    def ab_test_framework(self, test_name, control_segment, treatment_segment, metric='churned'):
        """Framework for A/B testing retention strategies"""
        control_data = self.data[self.data['customer_segment'].isin(control_segment)]
        treatment_data = self.data[self.data['customer_segment'].isin(treatment_segment)]
        
        # Calculate metrics
        control_metric = control_data[metric].mean()
        treatment_metric = treatment_data[metric].mean()
        
        # Statistical significance (simplified)
        n_control = len(control_data)
        n_treatment = len(treatment_data)
        
        # Effect size
        effect_size = treatment_metric - control_metric
        relative_change = (effect_size / control_metric) * 100 if control_metric > 0 else 0
        
        # Confidence interval (simplified calculation)
        pooled_std = np.sqrt(((control_metric * (1 - control_metric)) / n_control) + 
                            ((treatment_metric * (1 - treatment_metric)) / n_treatment))
        margin_of_error = 1.96 * pooled_std  # 95% confidence
        
        return {
            'test_name': test_name,
            'control_segments': control_segment,
            'treatment_segments': treatment_segment,
            'metric': metric,
            'control_value': control_metric,
            'treatment_value': treatment_metric,
            'effect_size': effect_size,
            'relative_change_percent': relative_change,
            'confidence_interval': (effect_size - margin_of_error, effect_size + margin_of_error),
            'sample_sizes': {'control': n_control, 'treatment': n_treatment}
        }
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        report = []
        
        # Overall metrics
        total_customers = len(self.data)
        overall_churn_rate = self.data['churned'].mean()
        avg_clv = self.data['customer_lifetime_value'].mean()
        
        report.append(f"=== FINN Customer Analytics Report ===\n")
        report.append(f"Total Customers Analyzed: {total_customers:,}")
        report.append(f"Overall Churn Rate: {overall_churn_rate:.1%}")
        report.append(f"Average Customer Lifetime Value: €{avg_clv:,.0f}\n")
        
        # Segment insights
        segment_analysis = self.customer_segmentation_analysis()
        report.append("=== Segment Performance ===")
        
        best_segment = segment_analysis['churned'].idxmin()
        worst_segment = segment_analysis['churned'].idxmax()
        highest_clv_segment = segment_analysis['customer_lifetime_value'].idxmax()
        
        report.append(f"Best Retention: {best_segment} ({segment_analysis.loc[best_segment, 'churned']:.1%} churn)")
        report.append(f"Highest Risk: {worst_segment} ({segment_analysis.loc[worst_segment, 'churned']:.1%} churn)")
        report.append(f"Highest CLV: {highest_clv_segment} (€{segment_analysis.loc[highest_clv_segment, 'customer_lifetime_value']:,.0f})\n")
        
        # At-risk customers
        if 'churn_probability' in self.data.columns:
            high_risk_customers = len(self.data[self.data['churn_probability'] > 0.7])
            at_risk_clv = self.data[self.data['churn_probability'] > 0.7]['customer_lifetime_value'].sum()
            
            report.append("=== Risk Assessment ===")
            report.append(f"High-Risk Customers (>70% churn probability): {high_risk_customers}")
            report.append(f"CLV at Risk: €{at_risk_clv:,.0f}\n")
        
        # Key factors
        if self.churn_model is not None:
            feature_importance = dict(zip(
                ['tenure_months', 'total_subscriptions', 'avg_subscription_length_months',
                 'avg_monthly_spend', 'vehicle_swaps', 'support_tickets_per_month',
                 'app_engagement_score', 'usage_intensity', 'price_sensitivity',
                 'subscriptions_per_year', 'swaps_per_subscription',
                 'high_engagement', 'payment_issues', 'is_premium_segment', 'is_price_sensitive',
                 'customer_segment_encoded', 'city_encoded', 'preferred_vehicle_type_encoded'],
                self.churn_model.feature_importances_
            ))
            
            top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            report.append("=== Top Churn Factors ===")
            for factor, importance in top_factors:
                report.append(f"- {factor}: {importance:.3f}")
        
        return "\n".join(report)

# Example usage and demonstration
def run_finn_analytics_demo():
    """Run complete analytics demonstration"""
    print(" FINN Customer Churn Analytics Demo")
    print("=" * 50)
    
    # Initialize analytics
    finn = FINNChurnAnalytics()
    
    # Generate and prepare data
    print(" Generating synthetic customer data...")
    data = finn.generate_customer_data(n_customers=5000)
    finn.prepare_churn_features()
    
    # Train churn model
    print(" Training churn prediction model...")
    model_results = finn.train_churn_model()
    
    # Analyze segments
    print("\n Customer Segment Analysis:")
    segment_analysis = finn.customer_segmentation_analysis()
    print(segment_analysis)
    
    # Identify at-risk customers
    print("\n Identifying at-risk customers...")
    at_risk = finn.identify_at_risk_customers(risk_threshold=0.6)
    print(f"Found {len(at_risk)} high-risk customers")
    print(at_risk.head())
    
    # Campaign optimization
    print("\n Retention Campaign Optimization:")
    campaigns = finn.retention_campaign_optimizer()
    print(campaigns)
    
    # A/B test example
    print("\n A/B Test Framework Example:")
    ab_result = finn.ab_test_framework(
        "Premium vs Standard Retention",
        control_segment=['Urban_Professional'],
        treatment_segment=['Premium_User']
    )
    print(f"Test: {ab_result['test_name']}")
    print(f"Effect Size: {ab_result['relative_change_percent']:.1f}%")
    
    # Generate insights report
    print("\n Executive Summary:")
    insights = finn.generate_insights_report()
    print(insights)
    
    # Create dashboard - THIS CREATES THE HTML FILE!
    print("\n Creating analytics dashboard...")
    dashboard = finn.create_analytics_dashboard()
    
    # Save data to CSV
    finn.data.to_csv('finn_customer_data_complete.csv', index=False)
    print("Customer data saved as: finn_customer_data_complete.csv")
    
    return finn, dashboard

if __name__ == "__main__":
    # Run the demonstration
    analytics, dashboard = run_finn_analytics_demo()
    
    # Show dashboard in browser (if possible)
    try:
        dashboard.show()
    except:
        print(" Note: Open finn_analytics_dashboard.html in your browser to view the dashboard")
    
    print("\n FINN Analytics Demo Complete!")
    print("Files created in your current directory:")
    print("   - finn_analytics_dashboard.html (Interactive Dashboard)")
    print("   - finn_customer_data_complete.csv (Complete Dataset)")
    print("\n This system demonstrates:")
    print("- Advanced customer segmentation")
    print("- Machine learning churn prediction") 
    print("- ROI-optimized retention campaigns")
    print("- A/B testing framework")
    print("- Comprehensive analytics dashboard")