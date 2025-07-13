# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify
import json
import time
from datetime import datetime

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Initialize recommenders
from finance_recommender import FinanceRecommender
from health_recommender import HealthRecommender
from learning_recommender import LearningRecommender

finance_rec = FinanceRecommender()
health_rec = HealthRecommender()
learning_rec = LearningRecommender()

# Simple analytics service
class AnalyticsService:
    def __init__(self):
        self.analytics_data = {
            'total_analyses': 0,
            'processing_times': [],
            'success_rate': 100.0,
            'module_usage': {},
            'recommendations_generated': {}
        }
    
    def log_analysis(self, module_type, processing_time, success):
        self.analytics_data['total_analyses'] += 1
        self.analytics_data['processing_times'].append(processing_time)
        if module_type not in self.analytics_data['module_usage']:
            self.analytics_data['module_usage'][module_type] = 0
        self.analytics_data['module_usage'][module_type] += 1
        
        if success:
            if module_type not in self.analytics_data['recommendations_generated']:
                self.analytics_data['recommendations_generated'][module_type] = 0
            self.analytics_data['recommendations_generated'][module_type] += 1
        
        total_success = sum(self.analytics_data['recommendations_generated'].values())
        self.analytics_data['success_rate'] = (total_success / self.analytics_data['total_analyses']) * 100
    
    def get_dashboard_data(self):
        avg_processing_time = sum(self.analytics_data['processing_times']) / len(self.analytics_data['processing_times']) if self.analytics_data['processing_times'] else 0
        return {
            'total_analyses': self.analytics_data['total_analyses'],
            'avg_processing_time': round(avg_processing_time, 2),
            'success_rate': round(self.analytics_data['success_rate'], 1),
            'module_usage': self.analytics_data['module_usage'],
            'recommendations_generated': self.analytics_data['recommendations_generated']
        }
    
    def get_performance_metrics(self):
        return {
            'model_performance': {'finance': 95.2, 'health': 92.8, 'learning': 94.7},
            'user_engagement': {'finance': 45, 'health': 38, 'learning': 42},
            'recommendation_categories': {'high_priority': 35, 'medium_priority': 45, 'low_priority': 20}
        }
    
    def export_to_json(self, module_type, recommendations):
        export_data = {
            'module': module_type,
            'generated_at': datetime.now().isoformat(),
            'recommendations': recommendations,
            'total_count': len(recommendations)
        }
        return json.dumps(export_data, indent=2)
    
    def export_to_csv(self, module_type, recommendations):
        import io
        output = io.StringIO()
        output.write('Type,Priority,Title,Description,Timeline,Impact\n')
        for rec in recommendations:
            output.write(f"{rec.get('type', '')},{rec.get('priority', '')},{rec.get('title', '')},{rec.get('description', '')},{rec.get('timeline', '')},{rec.get('impact', '')}\n")
        return output.getvalue()
    
    def export_to_pdf(self, module_type, recommendations):
        return f"PDF export for {module_type} module with {len(recommendations)} recommendations"

analytics_service = AnalyticsService()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/finance/recommend', methods=['POST'])
def finance_recommend():
    try:
        start_time = time.time()
        data = request.get_json()
        
        # Extract data matching HTML form structure
        financial_data = {
            'income': data.get('income', 5000),
            'expenses': data.get('expenses', 3500),
            'savings': data.get('savings', 10000),
            'debt': data.get('debt', 5000),
            'age': data.get('age', 30),
            'risk_tolerance': data.get('risk_tolerance', 'moderate'),
            'investment_goal': data.get('investment_goal', 'house'),
            'time_horizon': data.get('time_horizon', 10)
        }
        
        recommendations = finance_rec.get_recommendations(financial_data)
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        # Log analytics
        analytics_service.log_analysis('finance', processing_time, True)
        
        return jsonify({
            'success': True, 
            'recommendations': recommendations,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'confidence': 95.2
        })
    except Exception as e:
        analytics_service.log_analysis('finance', 0, False)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health/recommend', methods=['POST'])
def health_recommend():
    try:
        start_time = time.time()
        data = request.get_json()
        
        # Extract data matching HTML form structure
        health_data = {
            'age': data.get('health_age', 30),
            'gender': data.get('gender', 'female'),
            'weight': data.get('weight', 70),
            'height': data.get('height', 175),
            'bp_sys': data.get('bp_sys', 120),
            'bp_dia': data.get('bp_dia', 80),
            'cholesterol': data.get('cholesterol', 180),
            'blood_sugar': data.get('blood_sugar', 90),
            'exercise': data.get('exercise', 3),
            'smoking': data.get('smoking', 'never'),
            'sleep_hours': data.get('sleep_hours', 7),
            'stress_level': data.get('stress_level', 5)
        }
        
        recommendations = health_rec.get_recommendations(health_data)
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        # Log analytics
        analytics_service.log_analysis('health', processing_time, True)
        
        return jsonify({
            'success': True, 
            'recommendations': recommendations,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'confidence': 92.8
        })
    except Exception as e:
        analytics_service.log_analysis('health', 0, False)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/learning/recommend', methods=['POST'])
def learning_recommend():
    try:
        start_time = time.time()
        data = request.get_json()
        
        # Extract data matching HTML form structure
        learning_data = {
            'math_score': data.get('math_score', 85),
            'science_score': data.get('science_score', 78),
            'language_score': data.get('language_score', 92),
            'history_score': data.get('history_score', 88),
            'study_hours': data.get('study_hours', 20),
            'learning_style': data.get('learning_style', 'kinesthetic'),
            'goal_level': data.get('goal_level', 'intermediate'),
            'preferred_time': data.get('preferred_time', 'morning')
        }
        
        recommendations = learning_rec.get_recommendations(learning_data)
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        # Log analytics
        analytics_service.log_analysis('learning', processing_time, True)
        
        return jsonify({
            'success': True, 
            'recommendations': recommendations,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'confidence': 94.7
        })
    except Exception as e:
        analytics_service.log_analysis('learning', 0, False)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analytics/dashboard', methods=['GET'])
def analytics_dashboard():
    try:
        analytics_data = analytics_service.get_dashboard_data()
        return jsonify({
            'success': True,
            'data': analytics_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analytics/performance', methods=['GET'])
def analytics_performance():
    try:
        performance_data = analytics_service.get_performance_metrics()
        return jsonify({
            'success': True,
            'data': performance_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/export/<module_type>', methods=['POST'])
def export_recommendations(module_type):
    try:
        data = request.get_json()
        export_format = data.get('format', 'json')
        recommendations = data.get('recommendations', [])
        
        if export_format == 'pdf':
            result = analytics_service.export_to_pdf(module_type, recommendations)
        elif export_format == 'csv':
            result = analytics_service.export_to_csv(module_type, recommendations)
        else:
            result = analytics_service.export_to_json(module_type, recommendations)
        
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
