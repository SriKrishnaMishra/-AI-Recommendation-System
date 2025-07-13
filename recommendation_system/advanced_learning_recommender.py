import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
from datetime import datetime, timedelta

class AdvancedLearningRecommender:
    def __init__(self):
        self.performance_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.difficulty_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.clustering_model = KMeans(n_clusters=4, random_state=42)
        self.scaler = StandardScaler()
        
        self.learning_styles = {
            'visual': {'video_weight': 0.4, 'text_weight': 0.2, 'interactive_weight': 0.4},
            'auditory': {'video_weight': 0.5, 'text_weight': 0.1, 'interactive_weight': 0.4},
            'kinesthetic': {'video_weight': 0.2, 'text_weight': 0.2, 'interactive_weight': 0.6},
            'mixed': {'video_weight': 0.33, 'text_weight': 0.33, 'interactive_weight': 0.34}
        }
        
        self.train_models()
    
    def generate_synthetic_data(self, n_samples=1000):
        np.random.seed(42)
        
        math_scores = np.random.normal(75, 15, n_samples)
        science_scores = np.random.normal(73, 18, n_samples)
        language_scores = np.random.normal(78, 12, n_samples)
        study_hours = np.random.exponential(15, n_samples)
        
        math_scores = np.clip(math_scores, 0, 100)
        science_scores = np.clip(science_scores, 0, 100)
        language_scores = np.clip(language_scores, 0, 100)
        study_hours = np.clip(study_hours, 1, 50)
        
        avg_scores = (math_scores + science_scores + language_scores) / 3
        
        performance_labels = []
        for score in avg_scores:
            if score >= 90:
                performance_labels.append('excellent')
            elif score >= 80:
                performance_labels.append('good')
            elif score >= 70:
                performance_labels.append('average')
            else:
                performance_labels.append('needs_improvement')
        
        difficulty_scores = []
        for i, score in enumerate(avg_scores):
            base_difficulty = score / 100
            difficulty = base_difficulty + (study_hours[i] / 100) * 0.2 + np.random.normal(0, 0.1)
            difficulty_scores.append(np.clip(difficulty, 0.1, 1.0))
        
        return {
            'features': np.column_stack([math_scores, science_scores, language_scores, study_hours]),
            'performance_labels': performance_labels,
            'difficulty_scores': difficulty_scores
        }
    
    def train_models(self):
        data = self.generate_synthetic_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            data['features'], data['performance_labels'], test_size=0.2, random_state=42
        )
        
        self.performance_model.fit(X_train, y_train)
        
        X_train_diff, X_test_diff, y_train_diff, y_test_diff = train_test_split(
            data['features'], data['difficulty_scores'], test_size=0.2, random_state=42
        )
        
        self.difficulty_model.fit(X_train_diff, y_train_diff)
        
        scaled_features = self.scaler.fit_transform(data['features'])
        self.clustering_model.fit(scaled_features)
        
        y_pred = self.performance_model.predict(X_test)
        self.model_accuracy = accuracy_score(y_test, y_pred)
    
    def get_advanced_recommendation(self, learning_data):
        try:
            math_score = float(learning_data.get('math_score', 0))
            science_score = float(learning_data.get('science_score', 0))
            language_score = float(learning_data.get('language_score', 0))
            study_hours = float(learning_data.get('study_hours', 0))
            learning_style = learning_data.get('learning_style', 'mixed')
            goal_level = learning_data.get('goal_level', 'intermediate')
            
            features = np.array([[math_score, science_score, language_score, study_hours]])
            
            performance_prediction = self.performance_model.predict(features)[0]
            performance_proba = self.performance_model.predict_proba(features)[0]
            difficulty_recommendation = self.difficulty_model.predict(features)[0]
            
            scaled_features = self.scaler.transform(features)
            learning_cluster = self.clustering_model.predict(scaled_features)[0]
            
            avg_score = (math_score + science_score + language_score) / 3
            score_variance = np.var([math_score, science_score, language_score])
            
            recommendations = self._generate_personalized_recommendations(
                math_score, science_score, language_score, study_hours,
                learning_style, goal_level, performance_prediction,
                difficulty_recommendation, learning_cluster, avg_score, score_variance
            )
            
            study_plan = self._generate_study_plan(
                math_score, science_score, language_score, study_hours,
                learning_style, goal_level
            )
            
            confidence = max(performance_proba) * 100
            
            return {
                'recommendations': recommendations,
                'study_plan': study_plan,
                'performance_prediction': performance_prediction,
                'difficulty_level': round(difficulty_recommendation, 2),
                'learning_cluster': int(learning_cluster),
                'confidence_score': round(confidence, 1),
                'model_accuracy': round(self.model_accuracy * 100, 1),
                'avg_score': round(avg_score, 1),
                'score_variance': round(score_variance, 1),
                'strengths': self._identify_strengths(math_score, science_score, language_score),
                'weaknesses': self._identify_weaknesses(math_score, science_score, language_score),
                'learning_analytics': self._generate_learning_analytics(learning_data)
            }
            
        except Exception as e:
            return {
                'error': f'Error generating recommendations: {str(e)}',
                'recommendations': [{'title': 'Error', 'description': 'Please check your input data and try again.', 'priority': 'high'}]
            }
    
    def _generate_personalized_recommendations(self, math_score, science_score, language_score,
                                             study_hours, learning_style, goal_level,
                                             performance_prediction, difficulty_recommendation,
                                             learning_cluster, avg_score, score_variance):
        recommendations = []
        
        if performance_prediction == 'excellent':
            recommendations.append({
                'title': '🌟 Excellence Maintenance',
                'description': 'You're performing exceptionally well! Focus on advanced topics and consider mentoring others.',
                'priority': 'high',
                'action_items': [
                    'Explore advanced concepts in your strongest subjects',
                    'Consider participating in academic competitions',
                    'Start a study group to help peers'
                ]
            })
        elif performance_prediction == 'good':
            recommendations.append({
                'title': '📈 Performance Enhancement',
                'description': 'Good progress! Focus on consistency and tackling challenging problems.',
                'priority': 'medium',
                'action_items': [
                    'Increase practice with complex problems',
                    'Set specific improvement targets',
                    'Review and strengthen fundamental concepts'
                ]
            })
        elif performance_prediction == 'average':
            recommendations.append({
                'title': '🎯 Targeted Improvement',
                'description': 'You have solid foundations. Focus on identifying and addressing knowledge gaps.',
                'priority': 'medium',
                'action_items': [
                    'Take diagnostic tests to identify weak areas',
                    'Create a structured study schedule',
                    'Seek additional practice materials'
                ]
            })
        else:
            recommendations.append({
                'title': '🚀 Foundation Building',
                'description': 'Focus on building strong fundamentals with consistent practice.',
                'priority': 'high',
                'action_items': [
                    'Review basic concepts thoroughly',
                    'Increase daily study time gradually',
                    'Consider getting a tutor or study partner'
                ]
            })
        
        subject_scores = {'Math': math_score, 'Science': science_score, 'Language': language_score}
        lowest_subject = min(subject_scores, key=subject_scores.get)
        
        if score_variance > 200:
            recommendations.append({
                'title': f'📚 Focus on {lowest_subject}',
                'description': f'Your {lowest_subject.lower()} score needs attention. Allocate more study time to this subject.',
                'priority': 'high',
                'action_items': [
                    f'Dedicate 40% of study time to {lowest_subject.lower()}',
                    f'Find additional {lowest_subject.lower()} practice resources',
                    f'Consider {lowest_subject.lower()}-specific study techniques'
                ]
            })
        
        recommendations.append({
            'title': f'🎨 {learning_style.title()} Learning Optimization',
            'description': f'Optimize your study methods for {learning_style} learning.',
            'priority': 'medium',
            'action_items': self._get_learning_style_recommendations(learning_style)
        })
        
        if study_hours < 10:
            recommendations.append({
                'title': '⏰ Increase Study Time',
                'description': 'Consider increasing your weekly study hours for better results.',
                'priority': 'medium',
                'action_items': [
                    'Gradually increase study time by 2-3 hours per week',
                    'Create a consistent daily study routine',
                    'Use time-blocking techniques'
                ]
            })
        elif study_hours > 35:
            recommendations.append({
                'title': '🧘 Study-Life Balance',
                'description': 'You're studying a lot! Ensure you're maintaining a healthy balance.',
                'priority': 'medium',
                'action_items': [
                    'Include regular breaks in your study schedule',
                    'Focus on study quality over quantity',
                    'Ensure adequate sleep and recreation time'
                ]
            })
        
        return recommendations
    
    def _generate_study_plan(self, math_score, science_score, language_score,
                           study_hours, learning_style, goal_level):
        total_hours = min(study_hours, 40)
        
        scores = [math_score, science_score, language_score]
        subjects = ['Math', 'Science', 'Language']
        
        min_score = min(scores)
        allocations = []
        
        for score in scores:
            if score == min_score:
                allocation = 0.4
            elif score == max(scores):
                allocation = 0.25
            else:
                allocation = 0.35
            allocations.append(allocation)
        
        total_allocation = sum(allocations)
        allocations = [a / total_allocation for a in allocations]
        
        study_plan = {
            'weekly_schedule': {},
            'daily_recommendations': {},
            'resource_suggestions': {}
        }
        
        for i, subject in enumerate(subjects):
            hours = round(total_hours * allocations[i], 1)
            study_plan['weekly_schedule'][subject] = f"{hours} hours"
            
            daily_hours = round(hours / 7, 1)
            study_plan['daily_recommendations'][subject] = {
                'daily_time': f"{daily_hours} hours",
                'focus_areas': self._get_subject_focus_areas(subject, scores[i]),
                'study_methods': self._get_study_methods(learning_style, subject)
            }
        
        return study_plan
    
    def _get_learning_style_recommendations(self, learning_style):
        recommendations = {
            'visual': [
                'Use mind maps and diagrams',
                'Watch educational videos',
                'Create colorful notes and charts',
                'Use flashcards with images'
            ],
            'auditory': [
                'Listen to educational podcasts',
                'Study with background music',
                'Read notes aloud',
                'Join study groups for discussions'
            ],
            'kinesthetic': [
                'Use hands-on experiments',
                'Take frequent study breaks',
                'Use physical models and manipulatives',
                'Study while walking or standing'
            ],
            'mixed': [
                'Combine multiple learning methods',
                'Alternate between visual and auditory resources',
                'Use interactive online tools',
                'Experiment with different techniques'
            ]
        }
        return recommendations.get(learning_style, recommendations['mixed'])
    
    def _get_subject_focus_areas(self, subject, score):
        focus_areas = {
            'Math': {
                'low': ['Basic arithmetic', 'Algebra fundamentals', 'Problem-solving strategies'],
                'medium': ['Advanced algebra', 'Geometry', 'Statistics basics'],
                'high': ['Calculus', 'Advanced statistics', 'Mathematical modeling']
            },
            'Science': {
                'low': ['Scientific method', 'Basic concepts', 'Laboratory skills'],
                'medium': ['Advanced theories', 'Experimental design', 'Data analysis'],
                'high': ['Research methods', 'Advanced topics', 'Scientific writing']
            },
            'Language': {
                'low': ['Grammar basics', 'Vocabulary building', 'Reading comprehension'],
                'medium': ['Writing skills', 'Literature analysis', 'Communication'],
                'high': ['Advanced writing', 'Critical analysis', 'Creative expression']
            }
        }
        
        if score < 70:
            level = 'low'
        elif score < 85:
            level = 'medium'
        else:
            level = 'high'
        
        return focus_areas.get(subject, {}).get(level, ['General improvement'])
    
    def _get_study_methods(self, learning_style, subject):
        methods = {
            'visual': ['Diagrams', 'Charts', 'Videos', 'Infographics'],
            'auditory': ['Lectures', 'Discussions', 'Audio books', 'Verbal explanations'],
            'kinesthetic': ['Hands-on practice', 'Experiments', 'Physical models', 'Interactive exercises'],
            'mixed': ['Varied approaches', 'Multi-modal resources', 'Interactive content', 'Flexible methods']
        }
        return methods.get(learning_style, methods['mixed'])
    
    def _identify_strengths(self, math_score, science_score, language_score):
        strengths = []
        if math_score >= 85:
            strengths.append('Mathematics')
        if science_score >= 85:
            strengths.append('Science')
        if language_score >= 85:
            strengths.append('Language')
        return strengths
    
    def _identify_weaknesses(self, math_score, science_score, language_score):
        weaknesses = []
        if math_score < 70:
            weaknesses.append('Mathematics')
        if science_score < 70:
            weaknesses.append('Science')
        if language_score < 70:
            weaknesses.append('Language')
        return weaknesses
    
    def _generate_learning_analytics(self, learning_data):
        analytics = {
            'timestamp': datetime.now().isoformat(),
            'input_data': learning_data
        }
        return analytics