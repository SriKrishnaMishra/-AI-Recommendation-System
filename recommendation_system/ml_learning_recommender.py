import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import json

class MLLearningRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        
        # Sample course database with enhanced features
        self.courses_db = [
            {
                'id': 1, 'title': 'Python for Beginners', 'description': 'Learn Python programming basics syntax variables functions',
                'category': 'programming', 'difficulty': 'beginner', 'duration': 40, 'rating': 4.5,
                'skills': ['python', 'programming', 'basics'], 'prerequisites': []
            },
            {
                'id': 2, 'title': 'Advanced Python Programming', 'description': 'Deep dive into Python advanced features decorators generators',
                'category': 'programming', 'difficulty': 'advanced', 'duration': 80, 'rating': 4.7,
                'skills': ['python', 'advanced', 'decorators'], 'prerequisites': ['python basics']
            },
            {
                'id': 3, 'title': 'Machine Learning Fundamentals', 'description': 'Introduction to ML algorithms supervised unsupervised learning',
                'category': 'machine_learning', 'difficulty': 'intermediate', 'duration': 100, 'rating': 4.6,
                'skills': ['machine_learning', 'algorithms', 'data_science'], 'prerequisites': ['python', 'statistics']
            },
            {
                'id': 4, 'title': 'Deep Learning with PyTorch', 'description': 'Neural networks deep learning computer vision natural language processing',
                'category': 'machine_learning', 'difficulty': 'advanced', 'duration': 120, 'rating': 4.8,
                'skills': ['deep_learning', 'pytorch', 'neural_networks'], 'prerequisites': ['machine_learning', 'python']
            },
            {
                'id': 5, 'title': 'Data Science with R', 'description': 'Statistical analysis data visualization using R programming language',
                'category': 'data_science', 'difficulty': 'intermediate', 'duration': 90, 'rating': 4.4,
                'skills': ['r', 'statistics', 'data_visualization'], 'prerequisites': ['statistics']
            },
            {
                'id': 6, 'title': 'Web Development with React', 'description': 'Frontend development React components state management hooks',
                'category': 'programming', 'difficulty': 'intermediate', 'duration': 70, 'rating': 4.5,
                'skills': ['react', 'javascript', 'web_development'], 'prerequisites': ['javascript', 'html', 'css']
            },
            {
                'id': 7, 'title': 'Business Analytics', 'description': 'Business intelligence data analysis decision making strategies',
                'category': 'business', 'difficulty': 'intermediate', 'duration': 60, 'rating': 4.3,
                'skills': ['analytics', 'business', 'decision_making'], 'prerequisites': ['statistics']
            },
            {
                'id': 8, 'title': 'UI/UX Design Principles', 'description': 'User interface user experience design thinking prototyping',
                'category': 'design', 'difficulty': 'beginner', 'duration': 50, 'rating': 4.6,
                'skills': ['ui_design', 'ux_design', 'prototyping'], 'prerequisites': []
            },
            {
                'id': 9, 'title': 'Database Design and SQL', 'description': 'Relational databases SQL queries database optimization',
                'category': 'programming', 'difficulty': 'intermediate', 'duration': 65, 'rating': 4.4,
                'skills': ['sql', 'database', 'data_modeling'], 'prerequisites': ['programming_basics']
            },
            {
                'id': 10, 'title': 'Cloud Computing with AWS', 'description': 'Amazon Web Services cloud infrastructure serverless computing',
                'category': 'programming', 'difficulty': 'advanced', 'duration': 110, 'rating': 4.7,
                'skills': ['aws', 'cloud', 'infrastructure'], 'prerequisites': ['programming', 'networking']
            }
        ]
        
        # Initialize the ML models
        self._train_models()
    
    def _train_models(self):
        """Train the recommendation models"""
        # Create course descriptions for vectorization
        descriptions = [course['description'] + ' ' + ' '.join(course['skills']) for course in self.courses_db]
        
        # Fit TF-IDF vectorizer
        self.course_vectors = self.vectorizer.fit_transform(descriptions)
        
        # Fit clustering model
        self.kmeans.fit(self.course_vectors.toarray())
        
        # Add cluster labels to courses
        for i, course in enumerate(self.courses_db):
            course['cluster'] = self.kmeans.labels_[i]
    
    def get_recommendations(self, user_profile):
        """Get ML-based recommendations for a user"""
        recommendations = {
            'content_based': self._content_based_recommendations(user_profile),
            'collaborative_filtering': self._collaborative_filtering(user_profile),
            'learning_path': self._generate_learning_path(user_profile),
            'personalized_features': self._get_personalized_features(user_profile),
            'difficulty_progression': self._suggest_difficulty_progression(user_profile)
        }
        
        return recommendations
    
    def _content_based_recommendations(self, user_profile):
        """Content-based filtering using TF-IDF and cosine similarity"""
        # Create user profile vector
        user_interests = ' '.join(user_profile.get('interests', []))
        user_skills = ' '.join(user_profile.get('previous_courses', []))
        user_text = user_interests + ' ' + user_skills
        
        if not user_text.strip():
            return self._get_popular_courses()
        
        # Vectorize user profile
        user_vector = self.vectorizer.transform([user_text])
        
        # Calculate similarities
        similarities = cosine_similarity(user_vector, self.course_vectors).flatten()
        
        # Get top recommendations
        top_indices = similarities.argsort()[-5:][::-1]
        
        recommendations = []
        for idx in top_indices:
            course = self.courses_db[idx].copy()
            course['similarity_score'] = float(similarities[idx])
            course['recommendation_reason'] = self._get_recommendation_reason(course, user_profile)
            recommendations.append(course)
        
        return recommendations
    
    def _collaborative_filtering(self, user_profile):
        """Simulate collaborative filtering using clustering"""
        # Find similar users based on interests and skill level
        user_cluster = self._get_user_cluster(user_profile)
        
        # Get courses from the same cluster
        cluster_courses = [course for course in self.courses_db if course['cluster'] == user_cluster]
        
        # Sort by rating and return top recommendations
        cluster_courses.sort(key=lambda x: x['rating'], reverse=True)
        
        recommendations = []
        for course in cluster_courses[:5]:
            course_copy = course.copy()
            course_copy['recommendation_reason'] = f"Users with similar interests highly rated this course (Rating: {course['rating']})"
            recommendations.append(course_copy)
        
        return recommendations
    
    def _get_user_cluster(self, user_profile):
        """Assign user to a cluster based on their profile"""
        user_interests = ' '.join(user_profile.get('interests', []))
        user_skills = ' '.join(user_profile.get('previous_courses', []))
        user_text = user_interests + ' ' + user_skills
        
        if not user_text.strip():
            return 0  # Default cluster
        
        user_vector = self.vectorizer.transform([user_text])
        cluster = self.kmeans.predict(user_vector.toarray())[0]
        
        return cluster
    
    def _generate_learning_path(self, user_profile):
        """Generate a personalized learning path"""
        interests = user_profile.get('interests', [])
        skill_level = user_profile.get('skill_level', 'beginner')
        previous_courses = user_profile.get('previous_courses', [])
        
        path = {
            'current_level': skill_level,
            'recommended_sequence': [],
            'estimated_duration': 0,
            'learning_objectives': []
        }
        
        # Filter courses based on interests
        relevant_courses = [
            course for course in self.courses_db
            if any(interest in course['skills'] or interest in course['category'] for interest in interests)
        ]
        
        # Sort by difficulty progression
        difficulty_order = {'beginner': 0, 'intermediate': 1, 'advanced': 2}
        relevant_courses.sort(key=lambda x: difficulty_order.get(x['difficulty'], 1))
        
        # Build learning path
        current_skills = set(previous_courses)
        
        for course in relevant_courses:
            # Check if prerequisites are met
            if self._prerequisites_met(course, current_skills):
                path['recommended_sequence'].append({
                    'course': course,
                    'order': len(path['recommended_sequence']) + 1,
                    'rationale': self._get_path_rationale(course, current_skills)
                })
                path['estimated_duration'] += course['duration']
                current_skills.update(course['skills'])
                
                if len(path['recommended_sequence']) >= 6:  # Limit to 6 courses
                    break
        
        # Generate learning objectives
        path['learning_objectives'] = self._generate_learning_objectives(path['recommended_sequence'])
        
        return path
    
    def _prerequisites_met(self, course, current_skills):
        """Check if course prerequisites are met"""
        prerequisites = course.get('prerequisites', [])
        return all(prereq in current_skills for prereq in prerequisites)
    
    def _get_path_rationale(self, course, current_skills):
        """Get rationale for including course in learning path"""
        reasons = []
        
        if course['difficulty'] == 'beginner':
            reasons.append("Great starting point for building fundamentals")
        elif course['difficulty'] == 'intermediate':
            reasons.append("Builds on your existing knowledge")
        else:
            reasons.append("Advanced topic to deepen expertise")
        
        if course['rating'] >= 4.5:
            reasons.append("highly rated by students")
        
        return "; ".join(reasons)
    
    def _generate_learning_objectives(self, sequence):
        """Generate learning objectives based on course sequence"""
        objectives = []
        
        for item in sequence:
            course = item['course']
            objective = f"Master {', '.join(course['skills'][:3])} through {course['title']}"
            objectives.append(objective)
        
        return objectives
    
    def _get_personalized_features(self, user_profile):
        """Get personalized features and recommendations"""
        features = {
            'learning_style_adaptations': self._get_learning_style_adaptations(user_profile),
            'time_management': self._get_time_management_suggestions(user_profile),
            'skill_gap_analysis': self._analyze_skill_gaps(user_profile),
            'motivation_boosters': self._get_motivation_boosters(user_profile)
        }
        
        return features
    
    def _get_learning_style_adaptations(self, user_profile):
        """Adapt recommendations based on learning style"""
        style = user_profile.get('learning_style', 'visual')
        
        adaptations = {
            'visual': [
                'Focus on courses with infographics and diagrams',
                'Recommend video-based learning platforms',
                'Suggest mind mapping for note-taking'
            ],
            'auditory': [
                'Prioritize courses with audio lectures',
                'Recommend discussion forums and study groups',
                'Suggest podcast supplements'
            ],
            'hands_on': [
                'Emphasize project-based learning',
                'Recommend coding bootcamps and labs',
                'Suggest building portfolio projects'
            ],
            'reading': [
                'Recommend text-based courses and documentation',
                'Suggest academic papers and books',
                'Emphasize written exercises and assignments'
            ]
        }
        
        return adaptations.get(style, adaptations['visual'])
    
    def _get_time_management_suggestions(self, user_profile):
        """Get time management suggestions based on availability"""
        time_available = user_profile.get('time_available', 'moderate')
        
        suggestions = {
            'low': [
                'Focus on micro-learning (15-30 min sessions)',
                'Prioritize essential concepts',
                'Use spaced repetition for retention'
            ],
            'moderate': [
                'Allocate 1-2 hours daily for learning',
                'Mix theory with practical exercises',
                'Set weekly learning goals'
            ],
            'high': [
                'Consider intensive courses or bootcamps',
                'Engage in multiple courses simultaneously',
                'Participate in hackathons and challenges'
            ]
        }
        
        return suggestions.get(time_available, suggestions['moderate'])
    
    def _analyze_skill_gaps(self, user_profile):
        """Analyze skill gaps based on interests and current skills"""
        interests = user_profile.get('interests', [])
        current_skills = set(user_profile.get('previous_courses', []))
        
        # Define skill requirements for each interest area
        skill_requirements = {
            'programming': ['python', 'javascript', 'algorithms', 'data_structures'],
            'data_science': ['statistics', 'python', 'machine_learning', 'data_visualization'],
            'machine_learning': ['python', 'statistics', 'linear_algebra', 'neural_networks'],
            'web_development': ['html', 'css', 'javascript', 'react', 'backend'],
            'business': ['analytics', 'finance', 'marketing', 'strategy']
        }
        
        gaps = {}
        for interest in interests:
            if interest in skill_requirements:
                required_skills = set(skill_requirements[interest])
                missing_skills = required_skills - current_skills
                gaps[interest] = list(missing_skills)
        
        return gaps
    
    def _get_motivation_boosters(self, user_profile):
        """Get personalized motivation boosters"""
        boosters = [
            'Set small, achievable daily goals',
            'Track your progress with a learning journal',
            'Join online communities related to your interests',
            'Celebrate milestones and completed courses',
            'Find a study buddy or accountability partner'
        ]
        
        # Add personalized boosters based on profile
        if user_profile.get('skill_level') == 'beginner':
            boosters.append('Remember that everyone starts somewhere - focus on progress, not perfection')
        
        if 'programming' in user_profile.get('interests', []):
            boosters.append('Build small projects to see immediate results of your learning')
        
        return boosters
    
    def _suggest_difficulty_progression(self, user_profile):
        """Suggest difficulty progression based on current level"""
        current_level = user_profile.get('skill_level', 'beginner')
        
        progression = {
            'beginner': {
                'current_focus': 'Build strong fundamentals',
                'next_level': 'intermediate',
                'timeline': '3-6 months',
                'key_milestones': ['Complete 2-3 beginner courses', 'Build first project', 'Understand basic concepts']
            },
            'intermediate': {
                'current_focus': 'Deepen knowledge and practice',
                'next_level': 'advanced',
                'timeline': '6-12 months',
                'key_milestones': ['Complete advanced projects', 'Contribute to open source', 'Mentor beginners']
            },
            'advanced': {
                'current_focus': 'Specialize and lead',
                'next_level': 'expert',
                'timeline': '12+ months',
                'key_milestones': ['Publish research/articles', 'Lead projects', 'Teach others']
            }
        }
        
        return progression.get(current_level, progression['beginner'])
    
    def _get_recommendation_reason(self, course, user_profile):
        """Get reason for recommending a specific course"""
        reasons = []
        
        # Check interest alignment
        user_interests = user_profile.get('interests', [])
        if any(interest in course['skills'] or interest in course['category'] for interest in user_interests):
            reasons.append("matches your interests")
        
        # Check skill level appropriateness
        if course['difficulty'] == user_profile.get('skill_level', 'beginner'):
            reasons.append("suitable for your current skill level")
        
        # Check rating
        if course['rating'] >= 4.5:
            reasons.append("highly rated by students")
        
        return "Recommended because it " + " and ".join(reasons) if reasons else "Popular course in your area of interest"
    
    def _get_popular_courses(self):
        """Get popular courses as fallback"""
        popular_courses = sorted(self.courses_db, key=lambda x: x['rating'], reverse=True)[:5]
        
        for course in popular_courses:
            course['recommendation_reason'] = f"Popular course with {course['rating']} rating"
        
        return popular_courses