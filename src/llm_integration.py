"""
LLM Integration for Customer Analytics Platform
Natural language insights and recommendations using GPT-4
"""

from openai import OpenAI
import pandas as pd
import json
from typing import Dict, List, Optional
import os

class CustomerInsightsLLM:
    """Generate natural language insights from customer data using LLM"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM with API key"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
        self.model = "gpt-4o-mini"  # Cost-effective model
    
    def generate_customer_summary(self, customer_data: pd.Series) -> str:
        """Generate natural language summary for a customer"""
        
        customer_context = f"""
        Customer Profile:
        - Recency: {customer_data['recency_days']} days since last order
        - Frequency: {customer_data['frequency']} total orders
        - Monetary: ${customer_data['monetary']:.2f} total spent
        - RFM Segment: {customer_data['RFM_segment']}
        - Customer Lifetime Value: ${customer_data['historical_clv']:.2f}
        - Predicted 1-Year CLV: ${customer_data['predicted_clv_1year']:.2f}
        - Churn Status: {'Churned' if customer_data['is_churned'] else 'Active'}
        - Average Review Score: {customer_data.get('avg_review_score', 'N/A')}
        """
        
        prompt = f"""As a customer analytics expert, provide a concise 2-3 sentence summary of this customer's profile and recommend one specific action.

{customer_context}

Provide:
1. Brief customer profile summary
2. One specific, actionable recommendation"""

        try:
            if not self.client:
                return self._generate_rule_based_summary(customer_data)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a customer analytics expert providing actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            return self._generate_rule_based_summary(customer_data)
    
    def generate_segment_insights(self, segment_data: pd.DataFrame, segment_name: str) -> str:
        """Generate insights for an entire customer segment"""
        
        segment_stats = f"""
        Segment: {segment_name}
        - Total Customers: {len(segment_data):,}
        - Average Recency: {segment_data['recency_days'].mean():.0f} days
        - Average Frequency: {segment_data['frequency'].mean():.1f} orders
        - Average Monetary: ${segment_data['monetary'].mean():.2f}
        - Churn Rate: {segment_data['is_churned'].mean()*100:.1f}%
        - Total CLV: ${segment_data['historical_clv'].sum():,.0f}
        """
        
        prompt = f"""Analyze this customer segment and provide strategic recommendations.

{segment_stats}

Provide:
1. Key characteristics of this segment (2 sentences)
2. Top 2 strategic recommendations
3. Expected business impact"""

        try:
            if not self.client:
                return self._generate_rule_based_segment_insights(segment_data, segment_name)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strategic customer analytics consultant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            return self._generate_rule_based_segment_insights(segment_data, segment_name)
    
    def answer_business_question(self, question: str, data_summary: Dict) -> str:
        """Answer business questions about customer data"""
        
        context = f"""
        Dataset Overview:
        - Total Customers: {data_summary.get('total_customers', 'N/A'):,}
        - Churn Rate: {data_summary.get('churn_rate', 'N/A'):.1f}%
        - Average CLV: ${data_summary.get('avg_clv', 'N/A'):,.0f}
        - High-Value Customers: {data_summary.get('high_value_count', 'N/A'):,}
        - Top Segment: {data_summary.get('top_segment', 'N/A')}
        """
        
        prompt = f"""You are a data scientist analyzing e-commerce customer data. Answer this business question based on the data.

{context}

Question: {question}

Provide a clear, data-driven answer with specific recommendations."""

        try:
            if not self.client:
                return "LLM functionality requires an OpenAI API key. Using rule-based responses."
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior data scientist providing business insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Unable to generate LLM response. Error: {str(e)}"
    
    def _generate_rule_based_summary(self, customer_data: pd.Series) -> str:
        """Fallback rule-based summary when LLM unavailable"""
        
        segment = customer_data['RFM_segment']
        recency = customer_data['recency_days']
        frequency = customer_data['frequency']
        monetary = customer_data['monetary']
        
        if segment == 'Champions':
            return f"🌟 This is a Champion customer with {frequency} orders totaling ${monetary:.0f}. They ordered {recency} days ago. **Recommendation:** Maintain VIP treatment with exclusive offers and early access to new products."
        
        elif recency > 180:
            return f"⚠️ This customer hasn't ordered in {recency} days and is at high churn risk. They previously made {frequency} orders worth ${monetary:.0f}. **Recommendation:** Launch immediate win-back campaign with personalized discount."
        
        elif segment == 'Recent Customers':
            return f"🆕 New customer who made their first order {recency} days ago (${monetary:.0f}). **Recommendation:** Send onboarding email series and second-purchase incentive to drive repeat behavior."
        
        else:
            return f"This customer is in the '{segment}' segment with {frequency} orders (${monetary:.0f}) and last ordered {recency} days ago. **Recommendation:** Engage with targeted campaign based on purchase history."
    
    def _generate_rule_based_segment_insights(self, segment_data: pd.DataFrame, segment_name: str) -> str:
        """Fallback rule-based segment insights"""
        
        avg_clv = segment_data['historical_clv'].mean()
        churn_rate = segment_data['is_churned'].mean() * 100
        
        if segment_name == 'Champions':
            return f"""**Champions Segment Analysis:**
            
These are your most valuable customers with average CLV of ${avg_clv:.0f} and {churn_rate:.1f}% churn rate.

**Recommendations:**
1. Create VIP loyalty program with exclusive benefits
2. Use them as beta testers for new products

**Expected Impact:** Retain 95%+ of this segment, driving ${segment_data['historical_clv'].sum():,.0f} in lifetime value."""
        
        elif churn_rate > 60:
            return f"""**High-Risk Segment: {segment_name}**
            
This segment has a concerning {churn_rate:.1f}% churn rate with ${avg_clv:.0f} average CLV.

**Recommendations:**
1. Launch aggressive win-back campaign with time-limited offers
2. Conduct exit surveys to understand pain points

**Expected Impact:** Recover 20-30% of churned customers, adding ${segment_data['historical_clv'].sum()*0.25:,.0f} in recovered revenue."""
        
        else:
            return f"""**{segment_name} Segment:**
            
{len(segment_data):,} customers with ${avg_clv:.0f} average CLV and {churn_rate:.1f}% churn.

**Recommendations:**
1. Personalized engagement based on purchase patterns
2. Monitor regularly for segment migration

**Expected Impact:** Maintain current value and upgrade 10-15% to higher segments."""