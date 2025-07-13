import random
import json
import sys
import os
from datetime import datetime, timedelta

# Add the ml_models directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_models'))

try:
    from learning_ml_model import learning_ml_model
    ML_AVAILABLE = True
    print("✅ Learning ML Model loaded successfully!")
except ImportError as e:
    print(f"⚠️ Learning ML Model not available: {e}")
    ML_AVAILABLE = False

class LearningRecommender:
    def __init__(self):
        self.courses = {
            'programming': {
                'beginner': [
                    {'title': 'Python for Beginners', 'platform': 'Codecademy', 'duration': '4 weeks', 'type': 'interactive'},
                    {'title': 'Introduction to HTML/CSS', 'platform': 'FreeCodeCamp', 'duration': '3 weeks', 'type': 'project'},
                    {'title': 'JavaScript Basics', 'platform': 'MDN Web Docs', 'duration': '5 weeks', 'type': 'tutorial'}
                ],
                'intermediate': [
                    {'title': 'Data Structures and Algorithms', 'platform': 'LeetCode', 'duration': '8 weeks', 'type': 'practice'},
                    {'title': 'React Development', 'platform': 'Udemy', 'duration': '6 weeks', 'type': 'project'},
                    {'title': 'Python Flask Web Development', 'platform': 'Coursera', 'duration': '7 weeks', 'type': 'hands_on'}
                ],
                'advanced': [
                    {'title': 'System Design', 'platform': 'Educative', 'duration': '10 weeks', 'type': 'conceptual'},
                    {'title': 'Advanced Python Programming', 'platform': 'Real Python', 'duration': '12 weeks', 'type': 'deep_dive'},
                    {'title': 'Full Stack Development', 'platform': 'The Odin Project', 'duration': '16 weeks', 'type': 'comprehensive'}
                ]
            },
            'data_science': {
                'beginner': [
                    {'title': 'Introduction to Data Science', 'platform': 'Kaggle Learn', 'duration': '4 weeks', 'type': 'interactive'},
                    {'title': 'Statistics Fundamentals', 'platform': 'Khan Academy', 'duration': '6 weeks', 'type': 'video'},
                    {'title': 'Excel for Data Analysis', 'platform': 'Microsoft Learn', 'duration': '3 weeks', 'type': 'hands_on'}
                ],
                'intermediate': [
                    {'title': 'Python for Data Analysis', 'platform': 'DataCamp', 'duration': '8 weeks', 'type': 'interactive'},
                    {'title': 'Machine Learning Basics', 'platform': 'Coursera', 'duration': '10 weeks', 'type': 'project'},
                    {'title': 'SQL for Data Science', 'platform': 'SQLBolt', 'duration': '4 weeks', 'type': 'practice'}
                ],
                'advanced': [
                    {'title': 'Deep Learning Specialization', 'platform': 'Coursera', 'duration': '16 weeks', 'type': 'comprehensive'},
                    {'title': 'Advanced Machine Learning', 'platform': 'edX', 'duration': '12 weeks', 'type': 'theoretical'},
                    {'title': 'Data Science Capstone', 'platform': 'Kaggle', 'duration': '8 weeks', 'type': 'project'}
                ]
            },
            'machine_learning': {
                'beginner': [
                    {'title': 'Machine Learning for Everyone', 'platform': 'Coursera', 'duration': '6 weeks', 'type': 'conceptual'},
                    {'title': 'Introduction to AI', 'platform': 'MIT OpenCourseWare', 'duration': '8 weeks', 'type': 'academic'},
                    {'title': 'ML Crash Course', 'platform': 'Google AI', 'duration': '4 weeks', 'type': 'tutorial'}
                ],
                'intermediate': [
                    {'title': 'Applied Machine Learning', 'platform': 'Udacity', 'duration': '10 weeks', 'type': 'project'},
                    {'title': 'Machine Learning with Python', 'platform': 'IBM', 'duration': '8 weeks', 'type': 'hands_on'},
                    {'title': 'Computer Vision Basics', 'platform': 'OpenCV', 'duration': '6 weeks', 'type': 'practical'}
                ],
                'advanced': [
                    {'title': 'Advanced Deep Learning', 'platform': 'Fast.ai', 'duration': '14 weeks', 'type': 'research'},
                    {'title': 'Natural Language Processing', 'platform': 'Hugging Face', 'duration': '12 weeks', 'type': 'specialized'},
                    {'title': 'MLOps and Production ML', 'platform': 'MLflow', 'duration': '10 weeks', 'type': 'industry'}
                ]
            },
            'business': {
                'beginner': [
                    {'title': 'Business Fundamentals', 'platform': 'Coursera', 'duration': '6 weeks', 'type': 'conceptual'},
                    {'title': 'Introduction to Marketing', 'platform': 'Google Digital Garage', 'duration': '4 weeks', 'type': 'practical'},
                    {'title': 'Basic Finance', 'platform': 'Khan Academy', 'duration': '5 weeks', 'type': 'video'}
                ],
                'intermediate': [
                    {'title': 'Digital Marketing Strategy', 'platform': 'Google Ads', 'duration': '8 weeks', 'type': 'certification'},
                    {'title': 'Project Management', 'platform': 'PMI', 'duration': '10 weeks', 'type': 'professional'},
                    {'title': 'Financial Analysis', 'platform': 'Wharton Online', 'duration': '7 weeks', 'type': 'analytical'}
                ],
                'advanced': [
                    {'title': 'MBA Essentials', 'platform': 'edX', 'duration': '16 weeks', 'type': 'comprehensive'},
                    {'title': 'Strategic Management', 'platform': 'Harvard Business School', 'duration': '12 weeks', 'type': 'case_study'},
                    {'title': 'Entrepreneurship', 'platform': 'Startup School', 'duration': '10 weeks', 'type': 'practical'}
                ]
            },
            'design': {
                'beginner': [
                    {'title': 'Design Principles', 'platform': 'Coursera', 'duration': '4 weeks', 'type': 'conceptual'},
                    {'title': 'Adobe Creative Suite Basics', 'platform': 'Adobe', 'duration': '6 weeks', 'type': 'software'},
                    {'title': 'Color Theory', 'platform': 'Skillshare', 'duration': '3 weeks', 'type': 'creative'}
                ],
                'intermediate': [
                    {'title': 'UI/UX Design', 'platform': 'Google UX Design', 'duration': '8 weeks', 'type': 'certification'},
                    {'title': 'Advanced Photoshop', 'platform': 'Adobe', 'duration': '7 weeks', 'type': 'software'},
                    {'title': 'Web Design', 'platform': 'Webflow University', 'duration': '6 weeks', 'type': 'practical'}
                ],
                'advanced': [
                    {'title': 'Design Systems', 'platform': 'Design+Research', 'duration': '10 weeks', 'type': 'systematic'},
                    {'title': 'Motion Graphics', 'platform': 'School of Motion', 'duration': '12 weeks', 'type': 'specialized'},
                    {'title': 'Design Leadership', 'platform': 'IDEO', 'duration': '8 weeks', 'type': 'management'}
                ]
            }
        }
        
        self.learning_paths = {
            'career_change': {
                'assessment': 'Take skills assessment and identify gaps',
                'foundation': 'Build fundamental knowledge in target area',
                'practical': 'Complete hands-on projects and portfolio',
                'networking': 'Connect with professionals in the field',
                'certification': 'Obtain relevant certifications'
            },
            'skill_enhancement': {
                'current_skills': 'Evaluate current skill level',
                'advanced_topics': 'Focus on advanced concepts',
                'specialization': 'Choose specific area to specialize',
                'industry_trends': 'Stay updated with latest trends',
                'mentorship': 'Find mentors in the field'
            },
            'hobby_learning': {
                'exploration': 'Explore different areas of interest',
                'project_based': 'Learn through fun projects',
                'community': 'Join learning communities',
                'flexible_pace': 'Learn at your own pace',
                'enjoyment': 'Focus on topics you find interesting'
            }
        }

    def get_recommendations(self, learning_data):
        """Generate AI-powered learning recommendations"""

        # Use ML model if available
        if ML_AVAILABLE:
            try:
                ml_predictions = learning_ml_model.predict(learning_data)
                return self._generate_ml_learning_recommendations(learning_data, ml_predictions)
            except Exception as e:
                print(f"⚠️ Learning ML prediction failed: {e}")
                # Fall back to rule-based system

        # Rule-based recommendations (fallback)
        return self._generate_rule_based_learning_recommendations(learning_data)

    def _generate_ml_learning_recommendations(self, data, ml_predictions):
        """Generate learning recommendations using ML predictions"""
        recommendations = []

        current_avg = ml_predictions['current_average']
        future_performance = ml_predictions['predicted_future_performance']
        recommended_style = ml_predictions['recommended_learning_style']
        improvement_potential = ml_predictions['improvement_potential']
        study_efficiency = ml_predictions['study_efficiency']

        # Performance Improvement Strategy
        if future_performance > current_avg:
            improvement = future_performance - current_avg
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': '📈 Performance Improvement Plan',
                'description': f'AI predicts {improvement:.1f} point improvement possible. Current: {current_avg:.1f}%, Target: {future_performance:.1f}%',
                'action_items': [
                    f'Focus on {recommended_style} learning methods',
                    f'Improvement potential: {improvement_potential:.1f} points',
                    'Implement personalized study schedule',
                    'Track progress weekly'
                ],
                'timeline': '3-6 months',
                'impact': 'High'
            })

        # Subject-Specific Recommendations
        subjects = ['math', 'science', 'language', 'history']
        scores = [data['math_score'], data['science_score'], data['language_score'], data['history_score']]

        # Find weakest subject
        weakest_idx = scores.index(min(scores))
        weakest_subject = subjects[weakest_idx]
        weakest_score = scores[weakest_idx]

        if weakest_score < current_avg - 10:
            recommendations.append({
                'type': 'subject_focus',
                'priority': 'high',
                'title': f'🎯 {weakest_subject.title()} Improvement Focus',
                'description': f'Your {weakest_subject} score ({weakest_score:.1f}%) is significantly below your average ({current_avg:.1f}%)',
                'action_items': [
                    f'Dedicate 40% of study time to {weakest_subject}',
                    f'Use {recommended_style} learning techniques',
                    'Seek additional resources or tutoring',
                    'Practice daily for 30 minutes minimum'
                ],
                'timeline': '2-4 months',
                'impact': 'High'
            })

        # Study Efficiency Optimization
        if study_efficiency < 3:  # Less than 3 points per hour
            recommendations.append({
                'type': 'efficiency',
                'priority': 'medium',
                'title': '⚡ Study Efficiency Enhancement',
                'description': f'Your study efficiency is {study_efficiency:.1f} points/hour. Target: >4 points/hour',
                'action_items': [
                    'Implement Pomodoro Technique (25min focus blocks)',
                    'Eliminate distractions during study time',
                    'Use active learning techniques',
                    'Take regular breaks to maintain focus'
                ],
                'timeline': '2-3 weeks',
                'impact': 'Medium'
            })

        # Learning Style Optimization
        style_tips = {
            'visual': [
                'Use mind maps and diagrams',
                'Create colorful notes and charts',
                'Watch educational videos',
                'Use flashcards with images'
            ],
            'auditory': [
                'Listen to educational podcasts',
                'Study with background music',
                'Read notes aloud',
                'Join study groups for discussion'
            ],
            'kinesthetic': [
                'Use hands-on activities',
                'Take frequent movement breaks',
                'Use physical manipulatives',
                'Study while walking or standing'
            ],
            'reading': [
                'Read extensively on topics',
                'Take detailed written notes',
                'Use textbooks as primary resource',
                'Create written summaries'
            ]
        }

        recommendations.append({
            'type': 'learning_style',
            'priority': 'medium',
            'title': f'🧠 {recommended_style.title()} Learning Optimization',
            'description': f'AI recommends {recommended_style} learning approach based on your profile',
            'action_items': style_tips.get(recommended_style, ['Use varied learning techniques']),
            'timeline': '1-2 weeks',
            'impact': 'Medium'
        })

        # Study Schedule Optimization
        study_hours = data.get('study_hours', 20)
        if study_hours > 35:
            recommendations.append({
                'type': 'schedule',
                'priority': 'medium',
                'title': '⏰ Study Schedule Balance',
                'description': f'You study {study_hours} hours/week. Consider optimizing for quality over quantity.',
                'action_items': [
                    'Reduce to 25-30 hours per week',
                    'Focus on high-impact study activities',
                    'Include adequate rest and recreation',
                    'Maintain consistent daily schedule'
                ],
                'timeline': '1-2 weeks',
                'impact': 'Medium'
            })
        elif study_hours < 15:
            recommendations.append({
                'type': 'schedule',
                'priority': 'high',
                'title': '📚 Increase Study Time',
                'description': f'You study {study_hours} hours/week. Consider increasing for better results.',
                'action_items': [
                    'Gradually increase to 20-25 hours per week',
                    'Create consistent daily study routine',
                    'Use time-blocking techniques',
                    'Eliminate time-wasting activities'
                ],
                'timeline': '2-3 weeks',
                'impact': 'High'
            })

        # Goal-Level Specific Recommendations
        goal_level = data.get('goal_level', 'intermediate')
        if goal_level == 'beginner':
            recommendations.append({
                'type': 'foundation',
                'priority': 'high',
                'title': '🏗️ Build Strong Foundation',
                'description': 'Focus on fundamental concepts before advancing to complex topics.',
                'action_items': [
                    'Master basic concepts thoroughly',
                    'Use beginner-friendly resources',
                    'Practice fundamental skills daily',
                    'Seek help when concepts are unclear'
                ],
                'timeline': '2-4 months',
                'impact': 'High'
            })
        elif goal_level == 'advanced':
            recommendations.append({
                'type': 'advanced',
                'priority': 'medium',
                'title': '🚀 Advanced Learning Strategies',
                'description': 'Challenge yourself with complex problems and real-world applications.',
                'action_items': [
                    'Tackle challenging practice problems',
                    'Explore real-world applications',
                    'Consider teaching others',
                    'Pursue independent research projects'
                ],
                'timeline': '3-6 months',
                'impact': 'Medium'
            })

        # Time-of-Day Optimization
        preferred_time = data.get('preferred_time', 'morning')
        time_tips = {
            'morning': 'Schedule difficult subjects in the morning when focus is highest',
            'afternoon': 'Use afternoon for review and practice sessions',
            'evening': 'Evening study works well for reading and memorization',
            'night': 'Night study can be effective but ensure adequate sleep'
        }

        recommendations.append({
            'type': 'timing',
            'priority': 'low',
            'title': f'🕐 Optimize {preferred_time.title()} Study Sessions',
            'description': f'Maximize your {preferred_time} study preference',
            'action_items': [
                time_tips[preferred_time],
                'Maintain consistent study schedule',
                'Align difficult subjects with peak energy',
                'Allow flexibility for different subjects'
            ],
            'timeline': '1 week',
            'impact': 'Low'
        })

        # AI Model Insights
        recommendations.append({
            'type': 'ai_insight',
            'priority': 'low',
            'title': '🤖 AI Learning Analysis',
            'description': f'Learning assessment completed with {ml_predictions["model_confidence"]:.1%} confidence.',
            'action_items': [
                f'Current average: {current_avg:.1f}%',
                f'Predicted improvement: {future_performance - current_avg:.1f} points',
                f'Study efficiency: {study_efficiency:.1f} points/hour',
                f'Learning cluster: {ml_predictions["learning_cluster"]}'
            ],
            'timeline': 'Ongoing',
            'impact': 'Low'
        })

        return recommendations

    def _generate_rule_based_learning_recommendations(self, data):
        """Generate learning recommendations using rule-based system (fallback)"""
        recommendations = []

        # Calculate average score
        avg_score = (data['math_score'] + data['science_score'] +
                    data['language_score'] + data['history_score']) / 4

        # Basic performance recommendation
        if avg_score < 70:
            recommendations.append({
                'type': 'improvement',
                'priority': 'high',
                'title': '📚 Academic Improvement Plan',
                'description': f'Your average score is {avg_score:.1f}%. Focus on fundamental improvement.',
                'action_items': [
                    'Review basic concepts in all subjects',
                    'Increase study time gradually',
                    'Seek help from teachers or tutors',
                    'Create structured study schedule'
                ],
                'timeline': '2-3 months',
                'impact': 'High'
            })

        # Study time recommendation
        study_hours = data.get('study_hours', 20)
        if study_hours < 15:
            recommendations.append({
                'type': 'time_management',
                'priority': 'medium',
                'title': '⏰ Increase Study Time',
                'description': f'You study {study_hours} hours/week. Consider increasing for better results.',
                'action_items': [
                    'Gradually increase to 20+ hours per week',
                    'Create daily study routine',
                    'Use time management techniques',
                    'Eliminate distractions'
                ],
                'timeline': '2-3 weeks',
                'impact': 'Medium'
            })

        return recommendations

    def _get_course_recommendations(self, interests, skill_level):
        recommendations = []
        
        for interest in interests:
            if interest in self.courses:
                if skill_level in self.courses[interest]:
                    recommendations.extend(self.courses[interest][skill_level])
        
        # If no specific courses found, provide general recommendations
        if not recommendations:
            recommendations = [
                {'title': 'Learning How to Learn', 'platform': 'Coursera', 'duration': '4 weeks', 'type': 'meta_learning'},
                {'title': 'Critical Thinking Skills', 'platform': 'edX', 'duration': '6 weeks', 'type': 'foundational'},
                {'title': 'Research Methods', 'platform': 'Khan Academy', 'duration': '5 weeks', 'type': 'academic'}
            ]
        
        return recommendations[:5]  # Return top 5 recommendations

    def _create_learning_path(self, interests, skill_level, time_available):
        path = {
            'phase_1': {'title': 'Foundation', 'duration': '4-6 weeks', 'focus': 'Basic concepts and terminology'},
            'phase_2': {'title': 'Skill Building', 'duration': '6-8 weeks', 'focus': 'Hands-on practice and projects'},
            'phase_3': {'title': 'Application', 'duration': '4-6 weeks', 'focus': 'Real-world projects and portfolio'},
            'phase_4': {'title': 'Mastery', 'duration': '6-8 weeks', 'focus': 'Advanced topics and specialization'}
        }
        
        # Adjust based on time available
        if time_available == 'low':
            for phase in path.values():
                phase['duration'] = phase['duration'].replace('6-8', '4-6').replace('4-6', '3-4')
        elif time_available == 'high':
            for phase in path.values():
                phase['duration'] = phase['duration'].replace('4-6', '6-8').replace('6-8', '8-12')
        
        # Add specific milestones
        path['milestones'] = [
            'Complete foundational course',
            'Build first project',
            'Create portfolio piece',
            'Obtain certification or complete capstone'
        ]
        
        return path

    def _create_study_plan(self, time_available, learning_style):
        time_mapping = {
            'low': {'daily': '30-45 min', 'weekly': '3-4 hours', 'sessions': '3-4 per week'},
            'moderate': {'daily': '45-60 min', 'weekly': '5-7 hours', 'sessions': '4-5 per week'},
            'high': {'daily': '60-90 min', 'weekly': '8-12 hours', 'sessions': '5-6 per week'}
        }
        
        base_plan = time_mapping.get(time_available, time_mapping['moderate'])
        
        # Customize based on learning style
        if learning_style == 'visual':
            base_plan['techniques'] = ['Video lectures', 'Infographics', 'Mind maps', 'Diagrams']
            base_plan['tools'] = ['YouTube', 'Canva', 'MindMeister', 'Draw.io']
        elif learning_style == 'hands_on':
            base_plan['techniques'] = ['Practical exercises', 'Labs', 'Projects', 'Simulations']
            base_plan['tools'] = ['Codepen', 'Repl.it', 'GitHub', 'Jupyter Notebooks']
        elif learning_style == 'reading':
            base_plan['techniques'] = ['Textbooks', 'Articles', 'Documentation', 'Research papers']
            base_plan['tools'] = ['Kindle', 'Medium', 'arXiv', 'Google Scholar']
        else:  # auditory
            base_plan['techniques'] = ['Podcasts', 'Audio books', 'Discussions', 'Lectures']
            base_plan['tools'] = ['Audible', 'Spotify', 'Discord', 'Zoom']
        
        return base_plan

    def _get_additional_resources(self, interests, learning_style):
        resources = {
            'communities': [],
            'tools': [],
            'books': [],
            'websites': [],
            'podcasts': []
        }
        
        # Add interest-specific resources
        for interest in interests:
            if interest == 'programming':
                resources['communities'].extend(['Stack Overflow', 'Reddit r/programming', 'GitHub'])
                resources['tools'].extend(['VS Code', 'Git', 'Docker'])
                resources['books'].extend(['Clean Code', 'Design Patterns', 'The Pragmatic Programmer'])
                resources['websites'].extend(['MDN Web Docs', 'W3Schools', 'LeetCode'])
                resources['podcasts'].extend(['Software Engineering Daily', 'CodeNewbie', 'The Changelog'])
            
            elif interest == 'data_science':
                resources['communities'].extend(['Kaggle', 'DataCamp Community', 'KDnuggets'])
                resources['tools'].extend(['Jupyter', 'Pandas', 'Scikit-learn'])
                resources['books'].extend(['Python for Data Analysis', 'The Data Science Handbook'])
                resources['websites'].extend(['Towards Data Science', 'Analytics Vidhya'])
                resources['podcasts'].extend(['Data Skeptic', 'Linear Digressions', 'The Data Exchange'])
            
            elif interest == 'machine_learning':
                resources['communities'].extend(['ML Twitter', 'Papers with Code', 'Distill'])
                resources['tools'].extend(['TensorFlow', 'PyTorch', 'Weights & Biases'])
                resources['books'].extend(['Hands-On Machine Learning', 'Pattern Recognition and Machine Learning'])
                resources['websites'].extend(['Machine Learning Mastery', 'Towards AI'])
                resources['podcasts'].extend(['TWIML AI', 'Practical AI', 'The AI Podcast'])
        
        # Remove duplicates
        for key in resources:
            resources[key] = list(set(resources[key]))
        
        return resources

    def _get_assessment_plan(self, interests, skill_level):
        assessment = {
            'initial_assessment': 'Take diagnostic test to determine current knowledge',
            'progress_tracking': 'Weekly quizzes and project reviews',
            'milestone_tests': 'Monthly comprehensive assessments',
            'final_evaluation': 'Capstone project or certification exam'
        }
        
        # Add specific assessments based on interests
        specific_assessments = []
        for interest in interests:
            if interest == 'programming':
                specific_assessments.extend([
                    'Coding challenges on LeetCode/HackerRank',
                    'Code review sessions',
                    'Build and deploy a complete application'
                ])
            elif interest == 'data_science':
                specific_assessments.extend([
                    'Kaggle competition participation',
                    'Data analysis case studies',
                    'Statistical analysis projects'
                ])
            elif interest == 'machine_learning':
                specific_assessments.extend([
                    'Implement algorithms from scratch',
                    'Model evaluation and comparison',
                    'Research paper implementation'
                ])
        
        assessment['specific_assessments'] = specific_assessments
        
        return assessment