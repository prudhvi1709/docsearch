// Import required libraries
import { unsafeHTML } from "https://cdn.jsdelivr.net/npm/lit-html@3/directives/unsafe-html.js";
import { Marked } from "https://cdn.jsdelivr.net/npm/marked@13/+esm";

// Initialize marked
const marked = new Marked();

// Worker URL
const WORKER_URL = 'https://docsearch-embedding.prudhvi-krovvidi.workers.dev';
        
// Initialize elements
const searchForm = document.getElementById('search-form');
const searchQuery = document.getElementById('searchQuery');
const searchBtn = document.getElementById('searchBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const summarySection = document.getElementById('summarySection');
const errorMessage = document.getElementById('errorMessage');
const summaryText = document.getElementById('summaryText');
const documentResults = document.getElementById('documentResults');
const expandBtn = document.getElementById('expandBtn');
const collapseBtn = document.getElementById('collapseBtn');
const followUpQuestions = document.getElementById('followUpQuestions');
const expandCollapseButtons = document.getElementById('expandCollapseButtons');

// Sample question click handlers
document.querySelectorAll('.question').forEach(button => {
    button.addEventListener('click', (e) => {
        e.preventDefault();
        const query = e.target.getAttribute('data-query') || e.target.textContent;
        searchQuery.value = query;
        performSearch();
    });
});

// Form submit handler
searchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    performSearch();
});

// Expand/Collapse handlers
expandBtn.addEventListener('click', () => {
    document.querySelectorAll('.document-item .collapse').forEach(collapse => {
        new bootstrap.Collapse(collapse, { show: true });
    });
});

collapseBtn.addEventListener('click', () => {
    document.querySelectorAll('.document-item .collapse.show').forEach(collapse => {
        new bootstrap.Collapse(collapse, { hide: true });
    });
});

async function performSearch() {
    const query = searchQuery.value.trim();
    const ndocs = 10;

    if (!query) {
        showError('Please enter a search query.');
        return;
    }

    showLoading();

    try {
        const response = await fetch(`${WORKER_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                q: query,
                ndocs: ndocs
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        displayResults(data);
    } catch (err) {
        console.error('Search error:', err);
        showError(`Search failed: ${err.message}`);
    }
}

function showLoading() {
    loadingSpinner.classList.remove('d-none');
    summarySection.classList.add('d-none');
    errorMessage.classList.add('d-none');
    documentResults.innerHTML = '';
    expandCollapseButtons.classList.add('d-none');
    searchBtn.disabled = true;
}

function hideLoading() {
    loadingSpinner.classList.add('d-none');
    searchBtn.disabled = false;
}

function showError(message) {
    hideLoading();
    document.getElementById('errorText').textContent = message;
    errorMessage.classList.remove('d-none');
    summarySection.classList.add('d-none');
    expandCollapseButtons.classList.add('d-none');
}

function displayResults(response) {
    hideLoading();
    errorMessage.classList.add('d-none');

    if (!response.documents || response.documents.length === 0) {
        showError('No documents found for your query.');
        return;
    }

    // Display summary
    if (response.summary && response.summary.text) {
        const markdownContent = response.summary.text;
        const htmlContent = marked.parse(markdownContent);
        summaryText.innerHTML = htmlContent;
    } else {
        summaryText.textContent = 'No summary available.';
    }

    // Generate follow-up questions
    const mockFollowUps = [
        "What are India's specific targets for renewable energy?",
        "How does India plan to promote climate justice globally?",
        "What role does technology play in India's climate strategy?"
    ];
    
    followUpQuestions.innerHTML = mockFollowUps.map(question => 
        `<li class="mb-2"><button class="btn btn-link btn-sm text-start p-0 question" type="button" data-query="${escapeHtml(question)}">${escapeHtml(question)}</button></li>`
    ).join('');

    // Re-attach event listeners to follow-up questions
    followUpQuestions.querySelectorAll('.question').forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            const query = e.target.getAttribute('data-query') || e.target.textContent;
            searchQuery.value = query;
            performSearch();
        });
    });

    // Display documents
    documentResults.innerHTML = response.documents.map((doc, index) => {
        const percentage = Math.round(doc.score * 100);
        const preview = doc.metadata.content_preview || 'No preview available';
        const title = getDocumentTitle(doc);
        
        return `
            <div class="document-item border rounded p-3 mb-3">
                <div class="d-flex align-items-start">
                    <span class="badge bg-success rounded-circle p-1 me-2 mt-1" style="width: 10px; height: 10px;">&nbsp;</span>
                    <div class="flex-grow-1">
                        <div class="text-info fw-normal mb-1 small">
                            ${escapeHtml(title)} 
                            <span class="badge bg-light text-dark border ms-1">(${percentage}%)</span>
                        </div>
                        <div class="text-muted small">
                            ${escapeHtml(preview.substring(0, 150))}${preview.length > 150 ? '...' : ''}
                        </div>
                        <div class="collapse mt-2" id="collapse-${index}">
                            <div class="border rounded p-2">
                                <small class="text-muted">${escapeHtml(preview)}</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    summarySection.classList.remove('d-none');
    expandCollapseButtons.classList.remove('d-none');
}

function getDocumentTitle(doc) {
    if (doc.metadata.content_preview) {
        const preview = doc.metadata.content_preview;
        const sentences = preview.split(/[.!?]+/);
        if (sentences.length > 0 && sentences[0].length > 10) {
            return sentences[0].trim().substring(0, 100) + (sentences[0].length > 100 ? '...' : '');
        }
    }
    return doc.metadata.filename || `Document ${doc.id}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-focus search input
searchQuery.focus();