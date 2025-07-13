// Chart utilities for the recommendation system

// Finance Charts
function createFinanceCharts(data, recommendations) {
    createFinanceOverviewChart(data);
    createFinanceProjectionChart(data);
}

function createFinanceOverviewChart(data) {
    const ctx = document.getElementById('financeChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (window.financeChart) {
        window.financeChart.destroy();
    }
    
    const monthlyData = {
        income: data.income,
        expenses: data.expenses,
        surplus: data.income - data.expenses,
        savings: data.savings / 12 // Monthly equivalent
    };
    
    window.financeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Income', 'Expenses', 'Monthly Surplus', 'Avg Monthly Savings'],
            datasets: [{
                label: 'Financial Overview ($)',
                data: [monthlyData.income, monthlyData.expenses, monthlyData.surplus, monthlyData.savings],
                backgroundColor: [
                    'rgba(39, 174, 96, 0.8)',
                    'rgba(231, 76, 60, 0.8)',
                    'rgba(52, 152, 219, 0.8)',
                    'rgba(155, 89, 182, 0.8)'
                ],
                borderColor: [
                    'rgba(39, 174, 96, 1)',
                    'rgba(231, 76, 60, 1)',
                    'rgba(52, 152, 219, 1)',
                    'rgba(155, 89, 182, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Financial Overview'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

function createFinanceProjectionChart(data) {
    const ctx = document.getElementById('financeProjectionChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (window.financeProjectionChart) {
        window.financeProjectionChart.destroy();
    }
    
    // Calculate projections
    const years = [];
    const projectedSavings = [];
    const currentSavings = data.savings;
    const monthlySavings = data.income - data.expenses;
    
    for (let i = 0; i <= data.time_horizon; i++) {
        years.push(new Date().getFullYear() + i);
        projectedSavings.push(currentSavings + (monthlySavings * 12 * i));
    }
    
    window.financeProjectionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: [{
                label: 'Projected Savings',
                data: projectedSavings,
                borderColor: 'rgba(52, 152, 219, 1)',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Savings Projection'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

// Health Charts
function createHealthCharts(data, recommendations) {
    createHealthMetricsChart(data);
    createHealthRiskChart(data);
}

function createHealthMetricsChart(data) {
    const ctx = document.getElementById('healthChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (window.healthChart) {
        window.healthChart.destroy();
    }
    
    // Calculate BMI
    const bmi = data.weight / Math.pow(data.height / 100, 2);
    
    window.healthChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['BMI Score', 'Blood Pressure', 'Cholesterol', 'Exercise Hours', 'Sleep Hours', 'Stress Level (Inverted)'],
            datasets: [{
                label: 'Health Metrics',
                data: [
                    Math.min(bmi / 25 * 100, 100), // BMI normalized to 100
                    Math.max(0, 100 - (data.bp_sys - 120) / 2), // BP score
                    Math.max(0, 100 - (data.cholesterol - 200) / 2), // Cholesterol score
                    Math.min(data.exercise / 5 * 100, 100), // Exercise score
                    Math.min(data.sleep_hours / 8 * 100, 100), // Sleep score
                    Math.max(0, 100 - data.stress_level * 10) // Stress score (inverted)
                ],
                backgroundColor: 'rgba(46, 204, 113, 0.2)',
                borderColor: 'rgba(46, 204, 113, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(46, 204, 113, 1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Health Metrics Overview'
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20
                    }
                }
            }
        }
    });
}

function createHealthRiskChart(data) {
    const ctx = document.getElementById('healthRiskChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (window.healthRiskChart) {
        window.healthRiskChart.destroy();
    }
    
    // Calculate risk factors
    const bmi = data.weight / Math.pow(data.height / 100, 2);
    const risks = {
        'Weight': bmi > 25 || bmi < 18.5 ? 'High' : 'Low',
        'Blood Pressure': data.bp_sys > 140 || data.bp_dia > 90 ? 'High' : 'Low',
        'Cholesterol': data.cholesterol > 240 ? 'High' : 'Low',
        'Exercise': data.exercise < 3 ? 'High' : 'Low',
        'Smoking': data.smoking === 'current' ? 'High' : 'Low',
        'Stress': data.stress_level > 6 ? 'High' : 'Low'
    };
    
    const highRiskCount = Object.values(risks).filter(risk => risk === 'High').length;
    const lowRiskCount = Object.values(risks).filter(risk => risk === 'Low').length;
    
    window.healthRiskChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk Factors', 'High Risk Factors'],
            datasets: [{
                data: [lowRiskCount, highRiskCount],
                backgroundColor: [
                    'rgba(46, 204, 113, 0.8)',
                    'rgba(231, 76, 60, 0.8)'
                ],
                borderColor: [
                    'rgba(46, 204, 113, 1)',
                    'rgba(231, 76, 60, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Health Risk Assessment'
                },
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Learning Charts
function createLearningCharts(data, recommendations) {
    createLearningScoresChart(data);
    createLearningProgressChart(data);
}

function createLearningScoresChart(data) {
    const ctx = document.getElementById('learningChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (window.learningChart) {
        window.learningChart.destroy();
    }
    
    const subjects = ['Math', 'Science', 'Language', 'History'];
    const scores = [data.math_score, data.science_score, data.language_score, data.history_score];
    const average = scores.reduce((a, b) => a + b, 0) / scores.length;
    
    window.learningChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: subjects,
            datasets: [{
                label: 'Current Scores',
                data: scores,
                backgroundColor: scores.map(score => 
                    score >= average ? 'rgba(46, 204, 113, 0.8)' : 'rgba(231, 76, 60, 0.8)'
                ),
                borderColor: scores.map(score => 
                    score >= average ? 'rgba(46, 204, 113, 1)' : 'rgba(231, 76, 60, 1)'
                ),
                borderWidth: 2
            }, {
                label: 'Average',
                data: new Array(subjects.length).fill(average),
                type: 'line',
                borderColor: 'rgba(52, 152, 219, 1)',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 2,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Subject Performance'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

function createLearningProgressChart(data) {
    const ctx = document.getElementById('learningProgressChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (window.learningProgressChart) {
        window.learningProgressChart.destroy();
    }
    
    // Simulate progress projection
    const weeks = [];
    const mathProgress = [];
    const scienceProgress = [];
    const languageProgress = [];
    const historyProgress = [];
    
    for (let i = 0; i <= 12; i++) {
        weeks.push(`Week ${i}`);
        mathProgress.push(Math.min(100, data.math_score + (i * 2)));
        scienceProgress.push(Math.min(100, data.science_score + (i * 2.5)));
        languageProgress.push(Math.min(100, data.language_score + (i * 1.5)));
        historyProgress.push(Math.min(100, data.history_score + (i * 2.2)));
    }
    
    window.learningProgressChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: weeks,
            datasets: [{
                label: 'Math',
                data: mathProgress,
                borderColor: 'rgba(231, 76, 60, 1)',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                borderWidth: 2,
                tension: 0.4
            }, {
                label: 'Science',
                data: scienceProgress,
                borderColor: 'rgba(46, 204, 113, 1)',
                backgroundColor: 'rgba(46, 204, 113, 0.1)',
                borderWidth: 2,
                tension: 0.4
            }, {
                label: 'Language',
                data: languageProgress,
                borderColor: 'rgba(52, 152, 219, 1)',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 2,
                tension: 0.4
            }, {
                label: 'History',
                data: historyProgress,
                borderColor: 'rgba(155, 89, 182, 1)',
                backgroundColor: 'rgba(155, 89, 182, 0.1)',
                borderWidth: 2,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Projected Learning Progress'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

// Analytics Charts
function createAnalyticsCharts(data) {
    createModelPerformanceChart(data.model_performance);
    createEngagementChart(data.user_engagement);
    createCategoriesChart(data.recommendation_categories);
}

function createModelPerformanceChart(data) {
    const ctx = document.getElementById('modelPerformanceChart');
    if (!ctx) return;
    
    if (window.modelPerformanceChart) {
        window.modelPerformanceChart.destroy();
    }
    
    window.modelPerformanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Finance AI', 'Health AI', 'Learning AI'],
            datasets: [{
                label: 'Model Accuracy (%)',
                data: [data.finance, data.health, data.learning],
                backgroundColor: [
                    'rgba(52, 152, 219, 0.8)',
                    'rgba(46, 204, 113, 0.8)',
                    'rgba(155, 89, 182, 0.8)'
                ],
                borderColor: [
                    'rgba(52, 152, 219, 1)',
                    'rgba(46, 204, 113, 1)',
                    'rgba(155, 89, 182, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'AI Model Performance'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

function createEngagementChart(data) {
    const ctx = document.getElementById('engagementChart');
    if (!ctx) return;
    
    if (window.engagementChart) {
        window.engagementChart.destroy();
    }
    
    window.engagementChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Finance', 'Health', 'Learning'],
            datasets: [{
                data: [data.finance, data.health, data.learning],
                backgroundColor: [
                    'rgba(52, 152, 219, 0.8)',
                    'rgba(46, 204, 113, 0.8)',
                    'rgba(155, 89, 182, 0.8)'
                ],
                borderColor: [
                    'rgba(52, 152, 219, 1)',
                    'rgba(46, 204, 113, 1)',
                    'rgba(155, 89, 182, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'User Engagement by Module'
                }
            }
        }
    });
}

function createCategoriesChart(data) {
    const ctx = document.getElementById('categoriesChart');
    if (!ctx) return;
    
    if (window.categoriesChart) {
        window.categoriesChart.destroy();
    }
    
    window.categoriesChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['High Priority', 'Medium Priority', 'Low Priority'],
            datasets: [{
                data: [data.high_priority, data.medium_priority, data.low_priority],
                backgroundColor: [
                    'rgba(231, 76, 60, 0.8)',
                    'rgba(243, 156, 18, 0.8)',
                    'rgba(46, 204, 113, 0.8)'
                ],
                borderColor: [
                    'rgba(231, 76, 60, 1)',
                    'rgba(243, 156, 18, 1)',
                    'rgba(46, 204, 113, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Recommendation Priority Distribution'
                }
            }
        }
    });
}
