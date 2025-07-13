// Main application JavaScript
let currentTab = 'finance';
let currentRecommendations = {};

// Tab switching functionality
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to selected tab button
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    currentTab = tabName;
    
    // Load analytics data if analytics tab is selected
    if (tabName === 'analytics') {
        loadAnalytics();
    }
}

// Finance recommendations
function generateFinanceRecommendations() {
    const form = document.getElementById('finance-form');
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numeric fields
    ['income', 'expenses', 'savings', 'debt', 'age', 'time_horizon'].forEach(field => {
        data[field] = parseFloat(data[field]) || 0;
    });
    
    showLoading();
    
    fetch('/api/finance/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        hideLoading();
        if (result.success) {
            displayRecommendations('finance', result.recommendations);
            updateTimestamp('finance', result.timestamp);
            currentRecommendations.finance = result.recommendations;
            createFinanceCharts(data, result.recommendations);
        } else {
            showError('Failed to generate finance recommendations: ' + result.error);
        }
    })
    .catch(error => {
        hideLoading();
        showError('Error: ' + error.message);
    });
}

// Health recommendations
function generateHealthRecommendations() {
    const form = document.getElementById('health-form');
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numeric fields
    ['health_age', 'weight', 'height', 'bp_sys', 'bp_dia', 'cholesterol', 'blood_sugar', 'exercise', 'sleep_hours', 'stress_level'].forEach(field => {
        data[field] = parseFloat(data[field]) || 0;
    });
    
    showLoading();
    
    fetch('/api/health/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        hideLoading();
        if (result.success) {
            displayRecommendations('health', result.recommendations);
            updateTimestamp('health', result.timestamp);
            currentRecommendations.health = result.recommendations;
            createHealthCharts(data, result.recommendations);
        } else {
            showError('Failed to generate health recommendations: ' + result.error);
        }
    })
    .catch(error => {
        hideLoading();
        showError('Error: ' + error.message);
    });
}

// Learning recommendations
function generateLearningRecommendations() {
    const form = document.getElementById('learning-form');
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numeric fields
    ['math_score', 'science_score', 'language_score', 'history_score', 'study_hours'].forEach(field => {
        data[field] = parseFloat(data[field]) || 0;
    });
    
    showLoading();
    
    fetch('/api/learning/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        hideLoading();
        if (result.success) {
            displayRecommendations('learning', result.recommendations);
            updateTimestamp('learning', result.timestamp);
            currentRecommendations.learning = result.recommendations;
            createLearningCharts(data, result.recommendations);
        } else {
            showError('Failed to generate learning recommendations: ' + result.error);
        }
    })
    .catch(error => {
        hideLoading();
        showError('Error: ' + error.message);
    });
}

// Display recommendations
function displayRecommendations(module, recommendations) {
    const container = document.getElementById(`${module}-recommendations`);
    const resultsDiv = document.getElementById(`${module}-results`);
    
    container.innerHTML = '';
    
    recommendations.forEach(rec => {
        const recElement = document.createElement('div');
        recElement.className = `recommendation-item priority-${rec.priority}`;
        
        let actionItemsHtml = '';
        if (rec.action_items && rec.action_items.length > 0) {
            actionItemsHtml = '<ul>' + rec.action_items.map(item => `<li>${item}</li>`).join('') + '</ul>';
        }
        
        recElement.innerHTML = `
            <div class="recommendation-title">${rec.title}</div>
            <div class="recommendation-desc">${rec.description}</div>
            ${actionItemsHtml}
            <div style="margin-top: 10px; font-size: 12px; color: #666;">
                <span>Timeline: ${rec.timeline}</span> | 
                <span>Impact: ${rec.impact}</span> | 
                <span>Priority: ${rec.priority}</span>
            </div>
        `;
        
        container.appendChild(recElement);
    });
    
    resultsDiv.style.display = 'block';
}

// Update timestamp
function updateTimestamp(module, timestamp) {
    const timestampElement = document.getElementById(`${module}-timestamp`);
    if (timestampElement) {
        const date = new Date(timestamp);
        timestampElement.textContent = `Generated: ${date.toLocaleString()}`;
    }
}

// Reset form
function resetForm(formId) {
    document.getElementById(formId).reset();
    const module = formId.replace('-form', '');
    document.getElementById(`${module}-results`).style.display = 'none';
}

// Export recommendations
function exportRecommendations(module) {
    if (!currentRecommendations[module]) {
        showError('No recommendations to export. Please generate recommendations first.');
        return;
    }
    
    // Show export modal
    document.getElementById('exportModal').style.display = 'block';
    window.currentExportModule = module;
}

// Export functions
function exportToPDF() {
    exportData('pdf');
}

function exportToJSON() {
    exportData('json');
}

function exportToCSV() {
    exportData('csv');
}

function exportData(format) {
    const module = window.currentExportModule;
    const recommendations = currentRecommendations[module];
    
    fetch(`/api/export/${module}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            format: format,
            recommendations: recommendations
        })
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            // Create download link
            const blob = new Blob([result.data], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${module}_recommendations.${format}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            closeModal();
            showSuccess(`${format.toUpperCase()} export completed successfully!`);
        } else {
            showError('Export failed: ' + result.error);
        }
    })
    .catch(error => {
        showError('Export error: ' + error.message);
    });
}

// Load analytics
function loadAnalytics() {
    fetch('/api/analytics/dashboard')
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                updateAnalyticsDashboard(result.data);
            }
        });
    
    fetch('/api/analytics/performance')
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                createAnalyticsCharts(result.data);
            }
        });
}

// Update analytics dashboard
function updateAnalyticsDashboard(data) {
    document.getElementById('total-analyses').textContent = data.total_analyses;
    document.getElementById('avg-processing-time').textContent = data.avg_processing_time + 'ms';
    document.getElementById('success-rate').textContent = data.success_rate + '%';
}

// Utility functions
function showLoading() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.add('show');

    // Animate progress bar
    const progressFill = overlay.querySelector('.progress-fill');
    if (progressFill) {
        progressFill.style.width = '0%';
        setTimeout(() => {
            progressFill.style.width = '100%';
        }, 100);
    }

    // Safety timeout to hide loading after 30 seconds
    setTimeout(() => {
        if (overlay.classList.contains('show')) {
            hideLoading();
            showError('Request timed out. Please try again.');
        }
    }, 30000);
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.remove('show');
}

function showError(message) {
    hideLoading(); // Make sure loading is hidden on error
    showToast(message, 'error');
}

function showSuccess(message) {
    showToast(message, 'success');
}

function showToast(message, type) {
    const toast = document.createElement('div');
    toast.className = `alert alert-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10001;
        padding: 15px;
        border-radius: 5px;
        max-width: 300px;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

function closeModal() {
    document.getElementById('exportModal').style.display = 'none';
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Close modal when clicking outside
    window.onclick = function(event) {
        const modal = document.getElementById('exportModal');
        if (event.target === modal) {
            closeModal();
        }
    };
    
    // Close modal with close button
    document.querySelector('.close').onclick = closeModal;
});

// Add CSS animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;
document.head.appendChild(style);
