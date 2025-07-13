// Recommendation Engine - Client-side utilities and enhancements

// Recommendation scoring and ranking
class RecommendationEngine {
    constructor() {
        this.userPreferences = this.loadUserPreferences();
        this.interactionHistory = this.loadInteractionHistory();
    }
    
    // Load user preferences from localStorage
    loadUserPreferences() {
        const saved = localStorage.getItem('userPreferences');
        return saved ? JSON.parse(saved) : {
            priorityWeights: { high: 1.0, medium: 0.7, low: 0.4 },
            categoryPreferences: {},
            dismissedRecommendations: []
        };
    }
    
    // Load interaction history
    loadInteractionHistory() {
        const saved = localStorage.getItem('interactionHistory');
        return saved ? JSON.parse(saved) : {
            clicks: {},
            implementations: {},
            dismissals: {},
            ratings: {}
        };
    }
    
    // Save user preferences
    saveUserPreferences() {
        localStorage.setItem('userPreferences', JSON.stringify(this.userPreferences));
    }
    
    // Save interaction history
    saveInteractionHistory() {
        localStorage.setItem('interactionHistory', JSON.stringify(this.interactionHistory));
    }
    
    // Rank recommendations based on user preferences and history
    rankRecommendations(recommendations, module) {
        return recommendations.map(rec => {
            const score = this.calculateRecommendationScore(rec, module);
            return { ...rec, score, personalizedRank: score };
        }).sort((a, b) => b.score - a.score);
    }
    
    // Calculate recommendation score
    calculateRecommendationScore(recommendation, module) {
        let score = 0;
        
        // Base priority score
        const priorityWeight = this.userPreferences.priorityWeights[recommendation.priority] || 0.5;
        score += priorityWeight * 100;
        
        // Category preference
        const categoryPref = this.userPreferences.categoryPreferences[recommendation.type] || 0.5;
        score += categoryPref * 50;
        
        // Historical interaction bonus
        const interactionBonus = this.getInteractionBonus(recommendation.type, module);
        score += interactionBonus;
        
        // Recency factor (newer recommendations get slight boost)
        score += Math.random() * 10; // Simulate recency
        
        // Penalty for dismissed recommendations
        if (this.userPreferences.dismissedRecommendations.includes(recommendation.type)) {
            score *= 0.7;
        }
        
        return Math.round(score * 10) / 10;
    }
    
    // Get interaction bonus based on history
    getInteractionBonus(type, module) {
        const clicks = this.interactionHistory.clicks[`${module}_${type}`] || 0;
        const implementations = this.interactionHistory.implementations[`${module}_${type}`] || 0;
        const ratings = this.interactionHistory.ratings[`${module}_${type}`] || 0;
        
        return (clicks * 2) + (implementations * 10) + (ratings * 5);
    }
    
    // Track user interaction
    trackInteraction(type, module, action, value = 1) {
        const key = `${module}_${type}`;
        
        if (!this.interactionHistory[action]) {
            this.interactionHistory[action] = {};
        }
        
        this.interactionHistory[action][key] = (this.interactionHistory[action][key] || 0) + value;
        this.saveInteractionHistory();
    }
    
    // Update category preference based on user actions
    updateCategoryPreference(category, action) {
        const current = this.userPreferences.categoryPreferences[category] || 0.5;
        let adjustment = 0;
        
        switch (action) {
            case 'implement':
                adjustment = 0.1;
                break;
            case 'dismiss':
                adjustment = -0.05;
                break;
            case 'rate_high':
                adjustment = 0.08;
                break;
            case 'rate_low':
                adjustment = -0.03;
                break;
        }
        
        this.userPreferences.categoryPreferences[category] = Math.max(0, Math.min(1, current + adjustment));
        this.saveUserPreferences();
    }
    
    // Dismiss recommendation
    dismissRecommendation(type) {
        if (!this.userPreferences.dismissedRecommendations.includes(type)) {
            this.userPreferences.dismissedRecommendations.push(type);
            this.saveUserPreferences();
        }
        this.updateCategoryPreference(type, 'dismiss');
    }
    
    // Get personalized insights
    getPersonalizedInsights(module, recommendations) {
        const insights = [];
        
        // Most important categories for user
        const topCategories = Object.entries(this.userPreferences.categoryPreferences)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 3)
            .map(([category]) => category);
        
        if (topCategories.length > 0) {
            insights.push({
                type: 'preference',
                title: 'Your Focus Areas',
                description: `You tend to prioritize: ${topCategories.join(', ')}`,
                icon: '🎯'
            });
        }
        
        // Implementation rate
        const totalImplementations = Object.values(this.interactionHistory.implementations || {}).reduce((a, b) => a + b, 0);
        const totalRecommendations = Object.values(this.interactionHistory.clicks || {}).reduce((a, b) => a + b, 0);
        
        if (totalRecommendations > 0) {
            const implementationRate = (totalImplementations / totalRecommendations * 100).toFixed(1);
            insights.push({
                type: 'performance',
                title: 'Implementation Rate',
                description: `You implement ${implementationRate}% of recommendations you view`,
                icon: '📊'
            });
        }
        
        // Streak information
        const streak = this.calculateStreak(module);
        if (streak > 0) {
            insights.push({
                type: 'streak',
                title: 'Current Streak',
                description: `${streak} days of consistent engagement`,
                icon: '🔥'
            });
        }
        
        return insights;
    }
    
    // Calculate engagement streak
    calculateStreak(module) {
        // Simplified streak calculation
        const today = new Date();
        const lastWeek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
        
        // Simulate streak based on interaction history
        const recentActivity = Object.keys(this.interactionHistory.clicks || {})
            .filter(key => key.startsWith(module))
            .length;
        
        return Math.min(recentActivity * 2, 14); // Max 14 day streak
    }
    
    // Generate smart suggestions based on patterns
    generateSmartSuggestions(module, currentData) {
        const suggestions = [];
        
        // Pattern-based suggestions
        if (module === 'finance') {
            if (currentData.expenses > currentData.income * 0.8) {
                suggestions.push({
                    type: 'pattern',
                    title: 'High Expense Ratio Detected',
                    description: 'Consider reviewing your budget categories',
                    action: 'View expense breakdown tools'
                });
            }
        }
        
        if (module === 'health') {
            if (currentData.exercise < 3) {
                suggestions.push({
                    type: 'pattern',
                    title: 'Low Activity Pattern',
                    description: 'Gradual increase in activity might be more sustainable',
                    action: 'Start with 10-minute daily walks'
                });
            }
        }
        
        if (module === 'learning') {
            const scores = [currentData.math_score, currentData.science_score, currentData.language_score, currentData.history_score];
            const variance = this.calculateVariance(scores);
            
            if (variance > 100) {
                suggestions.push({
                    type: 'pattern',
                    title: 'Uneven Performance Detected',
                    description: 'Focus on balancing your subject strengths',
                    action: 'Create a balanced study schedule'
                });
            }
        }
        
        return suggestions;
    }
    
    // Calculate variance for array of numbers
    calculateVariance(numbers) {
        const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
        const squaredDiffs = numbers.map(num => Math.pow(num - mean, 2));
        return squaredDiffs.reduce((a, b) => a + b, 0) / numbers.length;
    }
    
    // Get recommendation trends
    getRecommendationTrends(module) {
        const trends = {
            mostPopular: this.getMostPopularRecommendations(module),
            emerging: this.getEmergingRecommendations(module),
            seasonal: this.getSeasonalRecommendations(module)
        };
        
        return trends;
    }
    
    // Get most popular recommendations
    getMostPopularRecommendations(module) {
        const moduleClicks = Object.entries(this.interactionHistory.clicks || {})
            .filter(([key]) => key.startsWith(module))
            .sort(([,a], [,b]) => b - a)
            .slice(0, 3);
        
        return moduleClicks.map(([key, count]) => ({
            type: key.split('_')[1],
            popularity: count,
            trend: 'up'
        }));
    }
    
    // Get emerging recommendations
    getEmergingRecommendations(module) {
        // Simulate emerging trends
        const emergingTypes = {
            finance: ['Digital Banking', 'Crypto Investment', 'ESG Funds'],
            health: ['Mental Wellness', 'Telehealth', 'Wearable Integration'],
            learning: ['AI-Assisted Learning', 'Microlearning', 'VR Education']
        };
        
        return (emergingTypes[module] || []).map(type => ({
            type,
            growth: Math.floor(Math.random() * 50) + 20,
            trend: 'emerging'
        }));
    }
    
    // Get seasonal recommendations
    getSeasonalRecommendations(module) {
        const month = new Date().getMonth();
        const seasonal = {
            finance: month < 3 ? ['Tax Planning', 'Year-end Review'] : ['Summer Budgeting', 'Vacation Planning'],
            health: month < 3 ? ['Winter Wellness', 'Immune Support'] : ['Summer Fitness', 'Hydration'],
            learning: month < 3 ? ['New Year Goals', 'Skill Building'] : ['Summer Courses', 'Certification Prep']
        };
        
        return (seasonal[module] || []).map(type => ({
            type,
            season: month < 3 ? 'Winter/Spring' : 'Summer/Fall',
            relevance: 'high'
        }));
    }
}

// Initialize recommendation engine
const recommendationEngine = new RecommendationEngine();

// Enhanced recommendation display with personalization
function displayPersonalizedRecommendations(module, recommendations) {
    const rankedRecommendations = recommendationEngine.rankRecommendations(recommendations, module);
    const insights = recommendationEngine.getPersonalizedInsights(module, recommendations);
    
    // Display insights first
    if (insights.length > 0) {
        displayInsights(module, insights);
    }
    
    // Display ranked recommendations
    displayRecommendations(module, rankedRecommendations);
}

// Display personalized insights
function displayInsights(module, insights) {
    const container = document.getElementById(`${module}-recommendations`);
    
    const insightsDiv = document.createElement('div');
    insightsDiv.className = 'insights-container';
    insightsDiv.innerHTML = '<h4>💡 Personalized Insights</h4>';
    
    insights.forEach(insight => {
        const insightElement = document.createElement('div');
        insightElement.className = 'insight-item';
        insightElement.innerHTML = `
            <span class="insight-icon">${insight.icon}</span>
            <div class="insight-content">
                <strong>${insight.title}</strong>
                <p>${insight.description}</p>
            </div>
        `;
        insightsDiv.appendChild(insightElement);
    });
    
    container.appendChild(insightsDiv);
}

// Add interaction tracking to recommendations
function addInteractionTracking(module) {
    const recommendations = document.querySelectorAll(`#${module}-recommendations .recommendation-item`);
    
    recommendations.forEach((rec, index) => {
        // Track clicks
        rec.addEventListener('click', () => {
            const type = rec.querySelector('.recommendation-title').textContent;
            recommendationEngine.trackInteraction(type, module, 'clicks');
        });
        
        // Add action buttons
        const actionDiv = document.createElement('div');
        actionDiv.className = 'recommendation-actions';
        actionDiv.innerHTML = `
            <button class="btn-small btn-implement" onclick="implementRecommendation('${module}', ${index})">✓ Implement</button>
            <button class="btn-small btn-dismiss" onclick="dismissRecommendation('${module}', ${index})">✗ Dismiss</button>
            <button class="btn-small btn-rate" onclick="rateRecommendation('${module}', ${index})">⭐ Rate</button>
        `;
        rec.appendChild(actionDiv);
    });
}

// Implement recommendation
function implementRecommendation(module, index) {
    const rec = document.querySelectorAll(`#${module}-recommendations .recommendation-item`)[index];
    const type = rec.querySelector('.recommendation-title').textContent;
    
    recommendationEngine.trackInteraction(type, module, 'implementations');
    recommendationEngine.updateCategoryPreference(type, 'implement');
    
    rec.classList.add('implemented');
    showSuccess('Recommendation marked as implemented!');
}

// Dismiss recommendation
function dismissRecommendation(module, index) {
    const rec = document.querySelectorAll(`#${module}-recommendations .recommendation-item`)[index];
    const type = rec.querySelector('.recommendation-title').textContent;
    
    recommendationEngine.dismissRecommendation(type);
    
    rec.style.opacity = '0.5';
    rec.classList.add('dismissed');
    showToast('Recommendation dismissed', 'info');
}

// Rate recommendation
function rateRecommendation(module, index) {
    const rec = document.querySelectorAll(`#${module}-recommendations .recommendation-item`)[index];
    const type = rec.querySelector('.recommendation-title').textContent;
    
    const rating = prompt('Rate this recommendation (1-5 stars):');
    if (rating && rating >= 1 && rating <= 5) {
        const action = rating >= 4 ? 'rate_high' : 'rate_low';
        recommendationEngine.trackInteraction(type, module, 'ratings', parseInt(rating));
        recommendationEngine.updateCategoryPreference(type, action);
        
        showSuccess(`Thank you for rating! (${rating}/5 stars)`);
    }
}

// Export for global use
window.recommendationEngine = recommendationEngine;
window.displayPersonalizedRecommendations = displayPersonalizedRecommendations;
window.addInteractionTracking = addInteractionTracking;
window.implementRecommendation = implementRecommendation;
window.dismissRecommendation = dismissRecommendation;
window.rateRecommendation = rateRecommendation;
