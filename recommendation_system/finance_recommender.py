import random
import json
import sys
import os
from datetime import datetime, timedelta

# Add the ml_models directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_models'))

try:
    from finance_ml_model import finance_ml_model
    ML_AVAILABLE = True
    print("✅ Finance ML Model loaded successfully!")
except ImportError as e:
    print(f"⚠️ Finance ML Model not available: {e}")
    ML_AVAILABLE = False

class FinanceRecommender:
    def __init__(self):
        self.investment_products = {
            'conservative': {
                'bonds': {'risk': 'low', 'return': '3-5%', 'description': 'Government and corporate bonds'},
                'savings_account': {'risk': 'very_low', 'return': '1-2%', 'description': 'High-yield savings accounts'},
                'cds': {'risk': 'very_low', 'return': '2-4%', 'description': 'Certificates of deposit'},
                'treasury_bills': {'risk': 'very_low', 'return': '2-3%', 'description': 'Short-term government securities'}
            },
            'moderate': {
                'index_funds': {'risk': 'moderate', 'return': '6-8%', 'description': 'Diversified stock market index funds'},
                'etfs': {'risk': 'moderate', 'return': '5-9%', 'description': 'Exchange-traded funds'},
                'mutual_funds': {'risk': 'moderate', 'return': '6-10%', 'description': 'Professionally managed investment funds'},
                'real_estate': {'risk': 'moderate', 'return': '4-7%', 'description': 'Real estate investment trusts (REITs)'}
            },
            'aggressive': {
                'individual_stocks': {'risk': 'high', 'return': '8-15%', 'description': 'Individual company stocks'},
                'growth_stocks': {'risk': 'high', 'return': '10-20%', 'description': 'High-growth potential stocks'},
                'cryptocurrency': {'risk': 'very_high', 'return': '10-50%', 'description': 'Digital currencies (highly volatile)'},
                'startup_investments': {'risk': 'very_high', 'return': '0-100%', 'description': 'Angel investing and venture capital'}
            }
        }
        
        self.budgeting_advice = {
            'low_income': {
                'emergency_fund': 'Build $1,000 emergency fund first',
                'budgeting': 'Use 50/30/20 rule: 50% needs, 30% wants, 20% savings',
                'debt': 'Focus on high-interest debt first',
                'tools': ['Mint', 'YNAB', 'Personal Capital']
            },
            'medium_income': {
                'emergency_fund': 'Build 3-6 months of expenses',
                'budgeting': 'Track all expenses and optimize categories',
                'debt': 'Consider debt consolidation if beneficial',
                'tools': ['Personal Capital', 'Quicken', 'Tiller']
            },
            'high_income': {
                'emergency_fund': 'Maintain 6-12 months of expenses',
                'budgeting': 'Focus on tax optimization and wealth building',
                'debt': 'Consider strategic debt vs investment decisions',
                'tools': ['Personal Capital', 'Quicken Premier', 'Financial Advisor']
            }
        }

    def get_recommendations(self, financial_data):
        """Generate AI-powered financial recommendations"""

        # Use ML model if available
        if ML_AVAILABLE:
            try:
                ml_predictions = finance_ml_model.predict(financial_data)
                return self._generate_ml_recommendations(financial_data, ml_predictions)
            except Exception as e:
                print(f"⚠️ ML prediction failed: {e}")
                # Fall back to rule-based system

        # Rule-based recommendations (fallback)
        return self._generate_rule_based_recommendations(financial_data)

    def _generate_ml_recommendations(self, data, ml_predictions):
        """Generate recommendations using ML predictions"""
        recommendations = []

        # Investment recommendations based on ML predictions
        predicted_risk = ml_predictions['predicted_risk_tolerance']
        investment_score = ml_predictions['investment_readiness_score']
        savings_growth = ml_predictions['predicted_savings_growth']

        # High priority recommendations
        if investment_score > 70:
            recommendations.append({
                'type': 'investment',
                'priority': 'high',
                'title': f'🚀 Investment Opportunity Detected',
                'description': f'Based on your financial profile, you have a {investment_score:.1f}% investment readiness score. Consider diversified portfolio allocation.',
                'action_items': [
                    f'Allocate funds based on {predicted_risk} risk tolerance',
                    f'Expected annual savings growth: ${savings_growth:,.0f}',
                    'Consider tax-advantaged accounts first'
                ],
                'timeline': '1-2 months',
                'impact': 'High'
            })

        # Savings optimization
        monthly_surplus = data['income'] - data['expenses']
        if monthly_surplus > 0:
            recommendations.append({
                'type': 'savings',
                'priority': 'high',
                'title': '💰 Optimize Your Savings Strategy',
                'description': f'With ${monthly_surplus:,.0f} monthly surplus, you can significantly boost your financial future.',
                'action_items': [
                    f'Automate ${monthly_surplus * 0.5:,.0f} monthly savings',
                    f'Build emergency fund: ${data["expenses"] * 6:,.0f} target',
                    'Consider high-yield savings accounts'
                ],
                'timeline': 'Immediate',
                'impact': 'High'
            })

        # Debt management
        if data['debt'] > 0:
            debt_to_income = data['debt'] / data['income']
            if debt_to_income > 0.3:
                recommendations.append({
                    'type': 'debt',
                    'priority': 'high',
                    'title': '⚠️ Debt Optimization Required',
                    'description': f'Your debt-to-income ratio is {debt_to_income:.1%}. Focus on debt reduction strategy.',
                    'action_items': [
                        'List all debts by interest rate',
                        'Consider debt consolidation options',
                        'Allocate extra payments to highest interest debt'
                    ],
                    'timeline': '3-6 months',
                    'impact': 'High'
                })

        # Age-specific recommendations
        if data['age'] < 35:
            recommendations.append({
                'type': 'retirement',
                'priority': 'medium',
                'title': '🎯 Early Career Investment Strategy',
                'description': 'Take advantage of compound interest with aggressive growth investments.',
                'action_items': [
                    'Maximize 401(k) employer match',
                    'Consider Roth IRA contributions',
                    'Focus on growth-oriented investments'
                ],
                'timeline': '3-6 months',
                'impact': 'Medium'
            })
        elif data['age'] > 50:
            recommendations.append({
                'type': 'retirement',
                'priority': 'high',
                'title': '🏁 Pre-Retirement Planning',
                'description': 'Focus on capital preservation and catch-up contributions.',
                'action_items': [
                    'Maximize catch-up contributions',
                    'Shift to more conservative investments',
                    'Plan for healthcare costs'
                ],
                'timeline': '1-3 months',
                'impact': 'High'
            })

        # Goal-specific recommendations
        if data.get('investment_goal') == 'house':
            recommendations.append({
                'type': 'goal',
                'priority': 'medium',
                'title': '🏠 Home Purchase Strategy',
                'description': 'Optimize your savings for home down payment.',
                'action_items': [
                    'Save 20% down payment to avoid PMI',
                    'Improve credit score for better rates',
                    'Consider first-time buyer programs'
                ],
                'timeline': f'{data.get("time_horizon", 5)} years',
                'impact': 'Medium'
            })

        # Add ML confidence and model info
        recommendations.append({
            'type': 'ai_insight',
            'priority': 'low',
            'title': '🤖 AI Model Insights',
            'description': f'Recommendations generated using advanced ML algorithms with {ml_predictions["model_confidence"]:.1%} confidence.',
            'action_items': [
                f'Risk tolerance prediction: {predicted_risk}',
                f'Investment readiness: {investment_score:.1f}/100',
                f'Projected savings growth: ${savings_growth:,.0f}/year'
            ],
            'timeline': 'Ongoing',
            'impact': 'Low'
        })

        return recommendations

    def _generate_rule_based_recommendations(self, data):
        """Generate recommendations using rule-based system (fallback)"""
        recommendations = []

        # Basic investment recommendation
        recommendations.append({
            'type': 'investment',
            'priority': 'medium',
            'title': '📈 Basic Investment Strategy',
            'description': 'Rule-based investment recommendations based on your profile.',
            'action_items': [
                f'Consider {data.get("risk_tolerance", "moderate")} risk investments',
                'Diversify your portfolio',
                'Review and rebalance quarterly'
            ],
            'timeline': '1-3 months',
            'impact': 'Medium'
        })

        # Basic savings recommendation
        monthly_surplus = data['income'] - data['expenses']
        if monthly_surplus > 0:
            recommendations.append({
                'type': 'savings',
                'priority': 'high',
                'title': '💰 Savings Opportunity',
                'description': f'You have ${monthly_surplus:,.0f} monthly surplus to optimize.',
                'action_items': [
                    'Set up automatic savings',
                    'Build emergency fund',
                    'Consider high-yield accounts'
                ],
                'timeline': 'Immediate',
                'impact': 'High'
            })

        return recommendations

    def _get_investment_recommendations(self, age, income, risk_tolerance):
        # Age-based asset allocation (100 - age = % in stocks)
        stock_percentage = max(20, min(90, 100 - age))
        bond_percentage = 100 - stock_percentage
        
        if risk_tolerance == 'conservative':
            risk_category = 'conservative'
            stock_percentage = max(20, stock_percentage - 20)
        elif risk_tolerance == 'aggressive':
            risk_category = 'aggressive'
            stock_percentage = min(90, stock_percentage + 20)
        else:
            risk_category = 'moderate'
        
        portfolio = {
            'asset_allocation': {
                'stocks': f"{stock_percentage}%",
                'bonds': f"{bond_percentage}%",
                'cash': "5-10%"
            },
            'recommended_products': self.investment_products[risk_category],
            'monthly_investment': self._calculate_monthly_investment(income)
        }
        return portfolio

    def _get_budgeting_advice(self, income):
        if income < 40000:
            category = 'low_income'
        elif income < 80000:
            category = 'medium_income'
        else:
            category = 'high_income'
        
        return self.budgeting_advice[category]

    def _get_goal_advice(self, goals, age, income):
        advice = {}
        
        for goal in goals:
            if goal == 'retirement':
                advice['retirement'] = {
                    'target_amount': income * 10,  # 10x annual income rule
                    'monthly_contribution': income * 0.15 / 12,  # 15% of income
                    'accounts': ['401k', 'IRA', 'Roth IRA'],
                    'timeline': 65 - age
                }
            elif goal == 'house':
                advice['house'] = {
                    'down_payment': income * 0.6,  # 20% down on 3x income house
                    'monthly_savings': income * 0.1 / 12,  # 10% of income
                    'timeline': '2-5 years',
                    'tips': ['Improve credit score', 'Research first-time buyer programs']
                }
            elif goal == 'education':
                advice['education'] = {
                    'target_amount': 50000,  # Average education cost
                    'monthly_savings': income * 0.05 / 12,  # 5% of income
                    'accounts': ['529 Plan', 'Coverdell ESA'],
                    'timeline': '5-18 years'
                }
            elif goal == 'emergency_fund':
                advice['emergency_fund'] = {
                    'target_amount': income * 0.5,  # 6 months expenses
                    'monthly_savings': income * 0.1 / 12,  # 10% of income
                    'accounts': ['High-yield savings', 'Money market'],
                    'timeline': '6-12 months'
                }
        
        return advice

    def _assess_risk_profile(self, age, income, risk_tolerance):
        risk_score = 0
        
        # Age factor (younger = higher risk capacity)
        if age < 30:
            risk_score += 3
        elif age < 50:
            risk_score += 2
        else:
            risk_score += 1
        
        # Income factor
        if income > 100000:
            risk_score += 3
        elif income > 50000:
            risk_score += 2
        else:
            risk_score += 1
        
        # Risk tolerance factor
        tolerance_scores = {'conservative': 1, 'moderate': 2, 'aggressive': 3}
        risk_score += tolerance_scores[risk_tolerance]
        
        if risk_score <= 3:
            profile = 'Conservative'
        elif risk_score <= 6:
            profile = 'Moderate'
        else:
            profile = 'Aggressive'
        
        return {
            'profile': profile,
            'score': risk_score,
            'description': self._get_risk_description(profile)
        }

    def _get_risk_description(self, profile):
        descriptions = {
            'Conservative': 'Focus on capital preservation with steady, low-risk returns',
            'Moderate': 'Balance growth and stability with moderate risk tolerance',
            'Aggressive': 'Prioritize growth with higher risk tolerance for potential higher returns'
        }
        return descriptions[profile]

    def _calculate_monthly_investment(self, income):
        return round(income * 0.15 / 12, 2)  # 15% of income annually

    def _create_action_plan(self, age, income, risk_tolerance, goals):
        plan = []
        
        # Step 1: Emergency fund
        plan.append({
            'step': 1,
            'action': 'Build emergency fund',
            'description': 'Save 3-6 months of expenses in high-yield savings',
            'priority': 'high'
        })
        
        # Step 2: Employer match
        plan.append({
            'step': 2,
            'action': 'Maximize employer 401k match',
            'description': 'Contribute enough to get full employer match',
            'priority': 'high'
        })
        
        # Step 3: High-interest debt
        plan.append({
            'step': 3,
            'action': 'Pay off high-interest debt',
            'description': 'Focus on credit cards and loans above 7% interest',
            'priority': 'high'
        })
        
        # Step 4: Long-term investing
        plan.append({
            'step': 4,
            'action': 'Start long-term investing',
            'description': 'Open investment accounts and begin regular contributions',
            'priority': 'medium'
        })
        
        # Step 5: Goal-specific savings
        if goals:
            plan.append({
                'step': 5,
                'action': 'Goal-specific savings',
                'description': f'Set up targeted savings for: {", ".join(goals)}',
                'priority': 'medium'
            })
        
        return plan