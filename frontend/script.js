class AnomalyDetectorApp {
    constructor() {
        this.apiBase = 'http://localhost:5000';
        this.initializeEventListeners();
        this.checkSystemStatus();
        this.loadCategories();
    }

    initializeEventListeners() {
        const fileInput = document.getElementById('videoFile');
        const uploadBox = document.getElementById('uploadBox');

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = '#00f2fe';
            uploadBox.style.background = '#f0f8ff';
        });

        uploadBox.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = '#4facfe';
            uploadBox.style.background = '';
        });

        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = '#4facfe';
            uploadBox.style.background = '';

            if (e.dataTransfer.files.length > 0) {
                this.handleFileUpload(e.dataTransfer.files[0]);
            }
        });
    }

    async handleFileUpload(file) {
        this.showLoading();
        this.hideError();
        this.hideResults();

        const formData = new FormData();
        formData.append('video', file);

        try {
            const response = await fetch(`${this.apiBase}/detect`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.status === 'success') {
                this.displayResults(result);
            } else {
                this.showError(result.error || 'Detection failed');
            }
        } catch (error) {
            this.showError('Network error: Could not connect to server');
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        const anomalyType = document.getElementById('anomalyType');
        const confidence = document.getElementById('confidence');
        const processingTime = document.getElementById('processingTime');
        const framesAnalyzed = document.getElementById('framesAnalyzed');

        // Update main result
        anomalyType.textContent = result.anomaly_type;
        confidence.textContent = `${Math.round(result.confidence * 100)}%`;
        processingTime.textContent = `${result.processing_time}s`;
        framesAnalyzed.textContent = result.frame_count;

        // Update probability bars
        this.updateProbabilityBars(result.all_probabilities);

        // Style based on anomaly type
        this.styleResultCard(result.anomaly_type, result.confidence);

        resultsSection.style.display = 'block';
    }

    updateProbabilityBars(probabilities) {
        const barsContainer = document.querySelector('.probability-bars');
        barsContainer.innerHTML = '';

        Object.entries(probabilities).forEach(([category, data]) => {
            const percentage = Math.round(data.probability * 100);
            const bar = `
                <div class="probability-bar" data-category="${category}">
                    <span class="category-name">${category.replace('_', ' ').toUpperCase()}</span>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${percentage}%"></div>
                        <span class="percentage">${percentage}%</span>
                    </div>
                </div>
            `;
            barsContainer.innerHTML += bar;
        });
    }

    styleResultCard(anomalyType, confidence) {
        const resultCard = document.querySelector('.result-card');
        const confidenceElement = document.querySelector('.confidence');

        // Reset styles
        resultCard.style.background = '#f8f9ff';
        confidenceElement.style.color = '#4caf50';

        if (anomalyType !== 'normal') {
            resultCard.style.background = '#fff0f0';
            confidenceElement.style.color = '#f44336';
        }

        // Add pulse animation for high confidence
        if (confidence > 0.8) {
            confidenceElement.style.animation = 'pulse 2s infinite';
        }
    }

    async checkSystemStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const status = await response.json();

            document.getElementById('modelStatus').textContent =
                status.model_loaded ? 'Loaded ✅' : 'Not Loaded ❌';
            document.getElementById('categoriesCount').textContent =
                status.categories ? status.categories.length : '0';
        } catch (error) {
            document.getElementById('modelStatus').textContent = 'Offline ❌';
        }
    }

    async loadCategories() {
        try {
            const response = await fetch(`${this.apiBase}/categories`);
            const categories = await response.json();
            console.log('Available categories:', categories);
        } catch (error) {
            console.error('Failed to load categories:', error);
        }
    }

    showLoading() {
        document.getElementById('loadingSection').style.display = 'block';
        document.getElementById('uploadBox').style.opacity = '0.5';
    }

    hideLoading() {
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('uploadBox').style.opacity = '1';
    }

    showError(message) {
        const errorSection = document.getElementById('errorSection');
        const errorText = document.getElementById('errorText');

        errorText.textContent = message;
        errorSection.style.display = 'block';
    }

    hideError() {
        document.getElementById('errorSection').style.display = 'none';
    }

    hideResults() {
        document.getElementById('resultsSection').style.display = 'none';
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new AnomalyDetectorApp();
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .confidence {
        animation: pulse 2s infinite;
    }

    .bar-fill {
        transition: width 1s ease-in-out;
    }
`;
document.head.appendChild(style);