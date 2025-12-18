document.addEventListener('DOMContentLoaded', function() {
    // Price slider
    const priceSlider = document.getElementById('price_slider');
    const priceValue = document.getElementById('price_value');
    const maxPriceInput = document.getElementById('max_price');
    
    if (priceSlider) {
        priceSlider.addEventListener('input', function() {
            const value = parseInt(this.value);
            priceValue.textContent = value.toLocaleString();
            maxPriceInput.value = value;
        });
    }
    
    // Load stats
    loadStats();
    
    // Form submission
    const form = document.getElementById('preference-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            getRecommendations();
        });
    }
});

function formatPrice(price) {
    return price.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.success) {
            const stats = data.stats;
            const container = document.getElementById('stats-container');
            
            container.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${stats.total_laptops}</div>
                    <div class="stat-label">Total Laptops</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.brands}</div>
                    <div class="stat-label">Brands</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${formatPrice(stats.avg_price)}</div>
                    <div class="stat-label">Avg Price</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${formatPrice(stats.min_price)}</div>
                    <div class="stat-label">Min Price</div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function getRecommendations() {
    const form = document.getElementById('preference-form');
    const formData = new FormData(form);
    
    const preferences = {
        ram: formData.get('ram'),
        max_price: formData.get('max_price'),
        storage: formData.get('storage'),
        display_size: formData.get('display_size')
    };
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    const resultsContainer = document.getElementById('results-container');
    resultsContainer.innerHTML = '';
    
    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(preferences)
        });
        
        const data = await response.json();
        
        // Hide loading
        document.getElementById('loading').style.display = 'none';
        
        if (data.success && data.recommendations.length > 0) {
            displayResults(data.recommendations);
        } else {
            resultsContainer.innerHTML = `
                <div class="no-results">
                    <i class="fas fa-search fa-3x"></i>
                    <h3>No laptops found</h3>
                    <p>Try adjusting your criteria</p>
                </div>
            `;
        }
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('loading').style.display = 'none';
        resultsContainer.innerHTML = `
            <div class="error">
                <i class="fas fa-exclamation-triangle fa-3x"></i>
                <h3>Error loading recommendations</h3>
                <p>${error.message}</p>
            </div>
        `;
    }
}

function displayResults(laptops) {
    const container = document.getElementById('results-container');
    
    laptops.forEach(laptop => {
        const scorePercent = Math.min(Math.round(laptop.score * 100), 100);
        
        const card = document.createElement('div');
        card.className = 'laptop-card';
        card.innerHTML = `
            <div class="laptop-header">
                <div>
                    <h3 class="laptop-name">${laptop.name}</h3>
                    <span class="laptop-brand">${laptop.brand}</span>
                </div>
                <div class="laptop-price">Rs ${formatPrice(laptop.price)}</div>
            </div>
            
            <div class="match-badge" style="
                background: ${scorePercent > 80 ? '#2ecc71' : scorePercent > 60 ? '#f39c12' : '#e74c3c'};
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                display: inline-block;
                margin-bottom: 10px;
                font-size: 0.9rem;
            ">
                ${scorePercent}% Match
            </div>
            
            <div class="laptop-specs">
                <div class="spec-item">
                    <i class="fas fa-memory"></i>
                    <div>
                        <div class="spec-label">RAM</div>
                        <div class="spec-value">${laptop.ram} GB</div>
                    </div>
                </div>
                <div class="spec-item">
                    <i class="fas fa-hdd"></i>
                    <div>
                        <div class="spec-label">Storage</div>
                        <div class="spec-value">${laptop.storage} GB</div>
                    </div>
                </div>
                <div class="spec-item">
                    <i class="fas fa-microchip"></i>
                    <div>
                        <div class="spec-label">Processor</div>
                        <div class="spec-value">${laptop.processor}</div>
                    </div>
                </div>
                <div class="spec-item">
                    <i class="fas fa-desktop"></i>
                    <div>
                        <div class="spec-label">Display</div>
                        <div class="spec-value">${laptop.display}"</div>
                    </div>
                </div>
            </div>
            
            <div class="laptop-actions">
                <a href="${laptop.url}" target="_blank" class="action-btn buy-btn">
                    <i class="fas fa-shopping-cart"></i> Buy Now
                </a>
            </div>
        `;
        
        container.appendChild(card);
    });
}
