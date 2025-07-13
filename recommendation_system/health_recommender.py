import random
import json
import sys
import os
from datetime import datetime, timedelta

# Add the ml_models directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_models'))

try:
    from health_ml_model import health_ml_model
    ML_AVAILABLE = True
    print("✅ Health ML Model loaded successfully!")
except ImportError as e:
    print(f"⚠️ Health ML Model not available: {e}")
    ML_AVAILABLE = False

class HealthRecommender:
    def __init__(self):
        self.exercise_plans = {
            'weight_loss': {
                'cardio': ['Running', 'Cycling', 'Swimming', 'HIIT'],
                'strength': ['Full body circuit', 'Compound movements'],
                'frequency': '5-6 days per week',
                'duration': '45-60 minutes'
            },
            'muscle_gain': {
                'cardio': ['Light cardio', 'Walking'],
                'strength': ['Progressive overload', 'Split training', 'Compound lifts'],
                'frequency': '4-5 days per week',
                'duration': '60-90 minutes'
            },
            'endurance': {
                'cardio': ['Long-distance running', 'Cycling', 'Swimming'],
                'strength': ['Functional training', 'Core strengthening'],
                'frequency': '5-6 days per week',
                'duration': '60-120 minutes'
            },
            'general_fitness': {
                'cardio': ['Brisk walking', 'Jogging', 'Dancing'],
                'strength': ['Bodyweight exercises', 'Light weights'],
                'frequency': '3-4 days per week',
                'duration': '30-45 minutes'
            }
        }
        
        self.nutrition_plans = {
            'weight_loss': {
                'calories': 'Caloric deficit of 500-750 calories per day',
                'macros': {'protein': '25-30%', 'carbs': '40-45%', 'fats': '25-30%'},
                'foods': ['Lean proteins', 'Vegetables', 'Whole grains', 'Fruits'],
                'avoid': ['Processed foods', 'Sugary drinks', 'Excessive fats']
            },
            'muscle_gain': {
                'calories': 'Caloric surplus of 300-500 calories per day',
                'macros': {'protein': '30-35%', 'carbs': '45-50%', 'fats': '20-25%'},
                'foods': ['Lean meats', 'Eggs', 'Dairy', 'Complex carbs', 'Healthy fats'],
                'avoid': ['Empty calories', 'Excessive alcohol', 'Processed foods']
            },
            'heart_health': {
                'calories': 'Maintain healthy weight',
                'macros': {'protein': '20-25%', 'carbs': '50-55%', 'fats': '25-30%'},
                'foods': ['Fish', 'Nuts', 'Olive oil', 'Fruits', 'Vegetables'],
                'avoid': ['Saturated fats', 'Trans fats', 'Excessive sodium']
            },
            'general_health': {
                'calories': 'Balanced intake for maintenance',
                'macros': {'protein': '20-25%', 'carbs': '45-50%', 'fats': '25-30%'},
                'foods': ['Variety of whole foods', 'Fruits', 'Vegetables', 'Whole grains'],
                'avoid': ['Processed foods', 'Excessive sugar', 'Alcohol']
            }
        }
        
        self.wellness_tips = {
            'stress_management': [
                'Practice meditation or mindfulness',
                'Get adequate sleep (7-9 hours)',
                'Exercise regularly',
                'Maintain social connections',
                'Consider professional counseling if needed'
            ],
            'sleep_hygiene': [
                'Stick to a consistent sleep schedule',
                'Create a relaxing bedtime routine',
                'Avoid screens before bedtime',
                'Keep bedroom cool and dark',
                'Limit caffeine and alcohol'
            ],
            'preventive_care': [
                'Regular health checkups',
                'Stay up to date with vaccinations',
                'Monitor blood pressure and cholesterol',
                'Cancer screenings as recommended',
                'Dental and vision checkups'
            ]
        }

    def get_recommendations(self, health_data):
        """Generate AI-powered health recommendations"""

        # Use ML model if available
        if ML_AVAILABLE:
            try:
                ml_predictions = health_ml_model.predict(health_data)
                return self._generate_ml_health_recommendations(health_data, ml_predictions)
            except Exception as e:
                print(f"⚠️ Health ML prediction failed: {e}")
                # Fall back to rule-based system

        # Rule-based recommendations (fallback)
        return self._generate_rule_based_health_recommendations(health_data)

    def _generate_ml_health_recommendations(self, data, ml_predictions):
        """Generate health recommendations using ML predictions"""
        recommendations = []

        current_bmi = ml_predictions['current_bmi']
        target_bmi = ml_predictions['target_bmi']
        health_score = ml_predictions['health_score']
        risk_level = ml_predictions['health_risk_level']
        bmi_category = ml_predictions['bmi_category']

        # BMI and Weight Management
        if abs(current_bmi - target_bmi) > 1:
            weight_change = (target_bmi - current_bmi) * ((data['height']/100) ** 2)
            action = "lose" if weight_change < 0 else "gain"

            recommendations.append({
                'type': 'weight_management',
                'priority': 'high' if abs(weight_change) > 10 else 'medium',
                'title': f'🎯 Weight Management Plan',
                'description': f'Your current BMI is {current_bmi:.1f} ({bmi_category}). Target BMI: {target_bmi:.1f}',
                'action_items': [
                    f'{action.title()} approximately {abs(weight_change):.1f} kg',
                    'Create sustainable caloric deficit/surplus',
                    'Combine diet and exercise approach',
                    'Track progress weekly'
                ],
                'timeline': f'{abs(weight_change)*4:.0f} weeks',
                'impact': 'High'
            })

        # Exercise Recommendations
        current_exercise = data.get('exercise', 0)
        if current_exercise < 3:
            recommendations.append({
                'type': 'exercise',
                'priority': 'high',
                'title': '🏃‍♀️ Increase Physical Activity',
                'description': f'You currently exercise {current_exercise} hours/week. Recommended: 5+ hours.',
                'action_items': [
                    'Start with 30 minutes daily walking',
                    'Add 2 strength training sessions per week',
                    'Include flexibility and balance exercises',
                    'Gradually increase intensity'
                ],
                'timeline': '2-4 weeks',
                'impact': 'High'
            })

        # Blood Pressure Management
        if data.get('bp_sys', 120) > 140:
            recommendations.append({
                'type': 'blood_pressure',
                'priority': 'high',
                'title': '⚠️ Blood Pressure Management',
                'description': f'Your systolic BP ({data["bp_sys"]} mmHg) is elevated. Target: <140 mmHg',
                'action_items': [
                    'Reduce sodium intake (<2300mg/day)',
                    'Increase potassium-rich foods',
                    'Regular cardiovascular exercise',
                    'Monitor BP daily',
                    'Consult healthcare provider'
                ],
                'timeline': '1-2 weeks',
                'impact': 'High'
            })

        # Cholesterol Management
        if data.get('cholesterol', 180) > 240:
            recommendations.append({
                'type': 'cholesterol',
                'priority': 'high',
                'title': '💓 Cholesterol Optimization',
                'description': f'Your cholesterol level ({data["cholesterol"]} mg/dL) needs attention. Target: <200 mg/dL',
                'action_items': [
                    'Increase fiber intake (25-35g/day)',
                    'Choose lean proteins and fish',
                    'Limit saturated and trans fats',
                    'Add plant sterols to diet'
                ],
                'timeline': '3-6 months',
                'impact': 'High'
            })

        # Sleep Optimization
        sleep_hours = data.get('sleep_hours', 7)
        if sleep_hours < 7 or sleep_hours > 9:
            recommendations.append({
                'type': 'sleep',
                'priority': 'medium',
                'title': '😴 Sleep Quality Improvement',
                'description': f'You sleep {sleep_hours} hours/night. Optimal range: 7-9 hours.',
                'action_items': [
                    'Establish consistent sleep schedule',
                    'Create relaxing bedtime routine',
                    'Limit screen time before bed',
                    'Optimize bedroom environment'
                ],
                'timeline': '2-3 weeks',
                'impact': 'Medium'
            })

        # Stress Management
        stress_level = data.get('stress_level', 5)
        if stress_level > 6:
            recommendations.append({
                'type': 'stress',
                'priority': 'medium',
                'title': '🧘‍♀️ Stress Reduction Strategy',
                'description': f'Your stress level ({stress_level}/10) is elevated. Target: <6/10',
                'action_items': [
                    'Practice daily meditation (10-15 minutes)',
                    'Try deep breathing exercises',
                    'Engage in regular physical activity',
                    'Consider professional counseling'
                ],
                'timeline': '1-4 weeks',
                'impact': 'Medium'
            })

        # Smoking Cessation
        if data.get('smoking') == 'current':
            recommendations.append({
                'type': 'smoking',
                'priority': 'high',
                'title': '🚭 Smoking Cessation Program',
                'description': 'Quitting smoking is the single best thing you can do for your health.',
                'action_items': [
                    'Set a quit date within 2 weeks',
                    'Consider nicotine replacement therapy',
                    'Join a support group',
                    'Identify and avoid triggers'
                ],
                'timeline': '2-12 weeks',
                'impact': 'High'
            })

        # Age-specific recommendations
        age = data.get('age', 30)
        if age > 40:
            recommendations.append({
                'type': 'preventive',
                'priority': 'medium',
                'title': '🔍 Preventive Health Screenings',
                'description': 'Age-appropriate health screenings are crucial for early detection.',
                'action_items': [
                    'Annual physical examination',
                    'Blood work (lipids, glucose, CBC)',
                    'Cancer screenings as recommended',
                    'Bone density test (if applicable)'
                ],
                'timeline': 'Annual',
                'impact': 'Medium'
            })

        # AI Model Insights
        recommendations.append({
            'type': 'ai_insight',
            'priority': 'low',
            'title': '🤖 AI Health Analysis',
            'description': f'Health assessment completed with {ml_predictions["model_confidence"]:.1%} confidence.',
            'action_items': [
                f'Overall health score: {health_score:.1f}/100',
                f'Risk level: {risk_level}',
                f'BMI category: {bmi_category}',
                'Recommendations personalized to your profile'
            ],
            'timeline': 'Ongoing',
            'impact': 'Low'
        })

        return recommendations

    def _generate_rule_based_health_recommendations(self, data):
        """Generate health recommendations using rule-based system (fallback)"""
        recommendations = []

        # Basic BMI recommendation
        bmi = data['weight'] / ((data['height'] / 100) ** 2)
        if bmi > 25:
            recommendations.append({
                'type': 'weight',
                'priority': 'high',
                'title': '⚖️ Weight Management',
                'description': f'Your BMI is {bmi:.1f}. Consider weight management strategies.',
                'action_items': [
                    'Consult with healthcare provider',
                    'Create balanced meal plan',
                    'Increase physical activity'
                ],
                'timeline': '3-6 months',
                'impact': 'High'
            })

        # Basic exercise recommendation
        if data.get('exercise', 0) < 3:
            recommendations.append({
                'type': 'exercise',
                'priority': 'medium',
                'title': '🏃‍♀️ Exercise Plan',
                'description': 'Increase your physical activity for better health.',
                'action_items': [
                    'Start with 30 minutes daily walking',
                    'Add strength training twice weekly',
                    'Include flexibility exercises'
                ],
                'timeline': '2-4 weeks',
                'impact': 'Medium'
            })

        return recommendations

    def _get_exercise_plan(self, activity_level, health_goals):
        plan = {}
        
        # Base plan on primary goal
        primary_goal = health_goals[0] if health_goals else 'general_fitness'
        
        if primary_goal in self.exercise_plans:
            plan = self.exercise_plans[primary_goal].copy()
        else:
            plan = self.exercise_plans['general_fitness'].copy()
        
        # Adjust for activity level
        if activity_level == 'low':
            plan['frequency'] = '2-3 days per week'
            plan['duration'] = '20-30 minutes'
            plan['intensity'] = 'Low to moderate'
        elif activity_level == 'high':
            plan['frequency'] = '6-7 days per week'
            plan['duration'] = '60-90 minutes'
            plan['intensity'] = 'Moderate to high'
        else:
            plan['intensity'] = 'Moderate'
        
        # Add progression plan
        plan['progression'] = {
            'week_1_2': 'Start with 50% intensity and duration',
            'week_3_4': 'Increase to 75% intensity and duration',
            'week_5_plus': 'Full intensity and duration'
        }
        
        return plan

    def _get_nutrition_plan(self, health_goals):
        primary_goal = health_goals[0] if health_goals else 'general_health'
        
        if primary_goal in self.nutrition_plans:
            base_plan = self.nutrition_plans[primary_goal].copy()
        else:
            base_plan = self.nutrition_plans['general_health'].copy()
        
        # Add general nutrition guidelines
        base_plan['hydration'] = 'Drink 8-10 glasses of water daily'
        base_plan['meal_timing'] = 'Eat 3 main meals and 2 healthy snacks'
        base_plan['supplements'] = ['Vitamin D', 'Omega-3', 'Multivitamin (if needed)']
        
        return base_plan

    def _get_wellness_tips(self, age, conditions):
        tips = []
        
        # Add age-specific tips
        if age < 30:
            tips.extend([
                'Establish healthy habits early',
                'Focus on building bone density',
                'Manage stress from life transitions'
            ])
        elif age < 50:
            tips.extend([
                'Monitor metabolic health',
                'Maintain muscle mass',
                'Balance work and family stress'
            ])
        else:
            tips.extend([
                'Focus on mobility and flexibility',
                'Monitor bone health',
                'Stay socially active'
            ])
        
        # Add condition-specific tips
        if 'diabetes' in conditions:
            tips.extend([
                'Monitor blood sugar regularly',
                'Focus on low glycemic index foods',
                'Regular foot care'
            ])
        
        if 'hypertension' in conditions:
            tips.extend([
                'Limit sodium intake',
                'Monitor blood pressure daily',
                'Practice stress reduction techniques'
            ])
        
        # Add general wellness tips
        tips.extend(self.wellness_tips['stress_management'])
        tips.extend(self.wellness_tips['sleep_hygiene'])
        
        return list(set(tips))  # Remove duplicates

    def _get_monitoring_recommendations(self, age, conditions):
        monitoring = {
            'daily': ['Weight', 'Energy levels', 'Sleep quality'],
            'weekly': ['Body measurements', 'Exercise performance'],
            'monthly': ['Progress photos', 'Fitness assessments'],
            'yearly': ['Complete physical exam', 'Blood work']
        }
        
        # Add age-specific monitoring
        if age >= 40:
            monitoring['yearly'].extend(['Cardiovascular screening', 'Cancer screenings'])
        
        if age >= 50:
            monitoring['yearly'].extend(['Bone density scan', 'Colonoscopy'])
        
        # Add condition-specific monitoring
        if 'diabetes' in conditions:
            monitoring['daily'].append('Blood glucose')
            monitoring['quarterly'] = ['HbA1c test']
        
        if 'hypertension' in conditions:
            monitoring['daily'].append('Blood pressure')
        
        return monitoring

    def _get_lifestyle_recommendations(self, age, activity_level, conditions):
        recommendations = []
        
        # General lifestyle recommendations
        recommendations.extend([
            'Maintain regular sleep schedule',
            'Limit alcohol consumption',
            'Avoid smoking and tobacco products',
            'Practice stress management techniques',
            'Stay hydrated throughout the day'
        ])
        
        # Activity level specific
        if activity_level == 'low':
            recommendations.extend([
                'Start with short walks',
                'Take stairs instead of elevators',
                'Set reminders to move every hour',
                'Find enjoyable physical activities'
            ])
        elif activity_level == 'high':
            recommendations.extend([
                'Ensure adequate recovery time',
                'Monitor for overtraining signs',
                'Maintain proper nutrition for training',
                'Consider working with a trainer'
            ])
        
        # Age-specific recommendations
        if age >= 65:
            recommendations.extend([
                'Focus on fall prevention',
                'Maintain social connections',
                'Regular vision and hearing checks',
                'Review medications regularly'
            ])
        
        # Condition-specific recommendations
        if conditions:
            recommendations.append('Follow medical provider recommendations')
            recommendations.append('Take medications as prescribed')
            recommendations.append('Monitor symptoms regularly')
        
        return recommendations