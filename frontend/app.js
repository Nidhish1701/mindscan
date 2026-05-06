/* ============================================================
   MindScan — Frontend JavaScript
   ============================================================ */

const API_BASE = window.location.origin;

// Chart.js global defaults
Chart.defaults.color = '#555a6b';
Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';
Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

let probChartInstance = null;
let labelChartInstance = null;
let trendChartInstance = null;
let lossChartInstance = null;
let profileChartInstance = null;
let breathInterval = null;

// ============================================================
// NAV
// ============================================================
const pageTitles = {
    analyzer:  ['Mood Check', 'Understand how your words reflect your emotional state'],
    dashboard: ['Overview', 'A summary of your recent emotional check-ins'],
    chatbot:   ['Support Chat', 'A safe space to share how you\'re feeling'],
    history:   ['My Reports', 'Your past mood check-ins and results'],
    profile:   ['Conversation Scan', 'Analyse multiple messages for emotional patterns over time'],
    coping:    ['Wellness Tools', 'Simple exercises to help you feel a little better'],
};

document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');

        const tab = btn.dataset.tab;
        document.getElementById(tab + 'Panel').classList.add('active');
        document.getElementById('pageTitle').textContent    = pageTitles[tab][0];
        document.getElementById('pageSubtitle').textContent = pageTitles[tab][1];

        if (tab === 'dashboard') loadDashboard();
        if (tab === 'history')   loadHistory();
    });
});

// ============================================================
// HEALTH CHECK
// ============================================================
async function checkHealth() {
    const dot  = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    try {
        const res  = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (data.model_loaded) {
            dot.className  = 'status-indicator online';
            text.textContent = 'Model Online';
        } else {
            dot.className  = 'status-indicator';
            text.textContent = 'Model Not Loaded';
        }
    } catch {
        dot.className  = 'status-indicator offline';
        text.textContent = 'API Offline';
    }
}
checkHealth();
setInterval(checkHealth, 20000);

// ============================================================
// ANALYZER
// ============================================================
const inputText = document.getElementById('inputText');
const charCount = document.getElementById('charCount');

inputText.addEventListener('input', () => {
    charCount.textContent = `${inputText.value.length} / 2000`;
    if (inputText.value.length > 2000) inputText.value = inputText.value.slice(0, 2000);
});

function clearInput() {
    inputText.value = '';
    charCount.textContent = '0 / 2000';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('analyzerEmpty').style.display  = '';
}

function setSample(text) {
    inputText.value = text;
    charCount.textContent = `${text.length} / 2000`;
}

async function analyzeText() {
    const text = inputText.value.trim();
    if (!text) { showToast('Please enter some text first.'); return; }

    const btn = document.getElementById('analyzeBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Analyzing…';

    try {
        const res = await fetch(`${API_BASE}/api/predict`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ text }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Analysis failed');
        }

        const data = await res.json();
        displayResults(data);
    } catch (err) {
        showToast('Error: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="m13 2-2 2.5h3L12 7"/><path d="M18 2h-8"/><path d="M22 12v7a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-7"/><path d="M22 12H2"/>
            </svg>
            Check my mood`;
    }
}

function displayResults(data) {
    document.getElementById('analyzerEmpty').style.display  = 'none';
    document.getElementById('resultsSection').style.display = '';

    // Header card
    const predLabel = document.getElementById('predLabel');
    predLabel.textContent = data.prediction;

    const ts = new Date().toLocaleTimeString();
    document.getElementById('predMeta').textContent = `Model: XGBoost · ${ts}`;

    // Severity badge
    const badge = document.getElementById('severityBadge');
    const severityStyles = {
        'Low':      { bg: 'rgba(52,211,153,0.12)',  color: '#34d399' },
        'Mild':     { bg: 'rgba(56,189,248,0.12)',  color: '#38bdf8' },
        'Moderate': { bg: 'rgba(251,191,36,0.12)',  color: '#fbbf24' },
        'High':     { bg: 'rgba(248,113,113,0.12)', color: '#f87171' },
        'Critical': { bg: 'rgba(251,113,133,0.18)', color: '#fb7185' },
    };
    const sty = severityStyles[data.severity] || severityStyles['Low'];
    badge.textContent        = data.is_crisis ? '⚠ CRISIS' : data.severity;
    badge.style.background   = data.is_crisis ? 'rgba(251,113,133,0.18)' : sty.bg;
    badge.style.color        = data.is_crisis ? '#fb7185' : sty.color;

    // Stats — round to whole number for natural feel
    document.getElementById('gaugeValue').textContent    = data.risk_score ?? '—';
    document.getElementById('predConfidence').textContent = `${Math.round(data.confidence * 100)}%`;
    document.getElementById('predLatency').textContent    = `${Math.round(data.latency_ms)}ms`;
    document.getElementById('predMeta').textContent = `Checked at ${new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}`;

    // Result header border accent
    const card = document.getElementById('resultHeaderCard');
    card.style.borderLeft = `3px solid ${sty.color}`;

    // Probability bars
    renderProbBars(data.probabilities, data.prediction);

    // Explanations
    renderExplanations(data.explanations);

    // Recommendations
    renderRecommendations(data.recommendations);
}

function renderProbBars(probs, topPred) {
    const container = document.getElementById('probBars');
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    const max = sorted[0][1];

    // Show only top 3 emotions for clean look
    const top3 = sorted.slice(0, 3);
    container.innerHTML = top3.map(([label, val]) => {
        const pct = Math.round(val * 100);
        const w   = ((val / max) * 100).toFixed(1);
        const isTop = label === topPred;
        // Soften clinical condition names
        const displayLabel = softConditionName(label);
        return `
            <div class="prob-row">
                <span class="prob-label${isTop ? '" style="color:var(--txt);font-weight:600' : ''}">${displayLabel}</span>
                <div class="prob-track"><div class="prob-fill${isTop ? ' top' : ''}" style="width:${w}%"></div></div>
                <span class="prob-pct">${pct}%</span>
            </div>`;
    }).join('');
}

function renderExplanations(explanations) {
    const container = document.getElementById('wordBars');
    if (!explanations || explanations.length === 0) {
        container.innerHTML = '<p class="empty-msg">No word-level explanations available.</p>';
        return;
    }

    const maxImp = Math.max(...explanations.map(e => e.importance));
    container.innerHTML = explanations.map(e => {
        const w       = Math.max(8, (e.importance / maxImp) * 100).toFixed(1);
        const isPos   = e.impact === 'supports';
        const impact  = isPos ? '↑ supports' : '↓ opposes';
        return `
            <div class="word-bar-row">
                <span class="word-name">${escapeHtml(e.word)}</span>
                <div class="word-track">
                    <div class="word-fill ${isPos ? 'pos' : 'neg'}" style="width:${w}%">${e.importance.toFixed(3)}</div>
                </div>
                <span class="word-impact">${impact}</span>
            </div>`;
    }).join('');
}

function renderRecommendations(recs) {
    const card      = document.getElementById('recommendationsCard');
    const container = document.getElementById('recommendationsList');
    if (!recs || recs.length === 0) { card.style.display = 'none'; return; }
    card.style.display = '';
    container.innerHTML = recs.map(r => `<div class="rec-item">${escapeHtml(r)}</div>`).join('');
}

// ============================================================
// DASHBOARD
// ============================================================
async function loadDashboard() {
    try {
        const res  = await fetch(`${API_BASE}/api/dashboard-data`);
        const data = await res.json();

        document.getElementById('kpiTotal').textContent      = data.total_predictions.toLocaleString();
        document.getElementById('kpiCrisis').textContent     = data.crisis_flags.toLocaleString();
        document.getElementById('kpiConfidence').textContent = `${Math.round(data.avg_confidence * 100)}%`;
        document.getElementById('kpiLatency').textContent    = `${Math.round(data.avg_latency_ms)}ms`;

        renderLabelChart(data.label_distribution);
        renderTrendChart(data.hourly_trend);
        renderRecentTable(data.recent_predictions);
    } catch {
        console.warn('Dashboard data unavailable');
    }
}

function renderLabelChart(dist) {
    const canvas = document.getElementById('labelChart');
    if (labelChartInstance) labelChartInstance.destroy();

    const labels = Object.keys(dist);
    const values = Object.values(dist);
    const colors = ['#5b8dee','#8b5cf6','#38bdf8','#34d399','#fbbf24','#f87171'];

    labelChartInstance = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors.slice(0, labels.length),
                borderColor: '#13161d',
                borderWidth: 3,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { padding: 16, usePointStyle: true, pointStyleWidth: 8, font: { size: 11 } } }
            }
        }
    });
}

function renderTrendChart(trend) {
    const canvas = document.getElementById('trendChart');
    if (trendChartInstance) trendChartInstance.destroy();

    if (!trend || trend.length === 0) {
        trendChartInstance = new Chart(canvas, {
            type: 'line',
            data: { labels: ['No data'], datasets: [{ data: [0], borderColor: '#5b8dee' }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
        return;
    }

    const labels = [...trend].reverse().map(h => h.hour);
    const counts = [...trend].reverse().map(h => h.count);
    const crisis = [...trend].reverse().map(h => h.crisis);

    trendChartInstance = new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Predictions',
                    data: counts,
                    borderColor: '#5b8dee',
                    backgroundColor: 'rgba(91,141,238,0.08)',
                    fill: true, tension: 0.4, borderWidth: 2,
                    pointRadius: 3, pointBackgroundColor: '#5b8dee',
                },
                {
                    label: 'Crisis',
                    data: crisis,
                    borderColor: '#f87171',
                    backgroundColor: 'rgba(248,113,113,0.06)',
                    fill: true, tension: 0.4, borderWidth: 2,
                    pointRadius: 3, pointBackgroundColor: '#f87171',
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { usePointStyle: true, pointStyleWidth: 8, font: { size: 11 } } } },
            scales: {
                x: { ticks: { maxTicksLimit: 8, maxRotation: 0, font: { size: 10 } } },
                y: { beginAtZero: true }
            }
        }
    });
}

function renderRecentTable(preds) {
    const tbody = document.getElementById('recentTableBody');
    if (!preds || preds.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-row">No predictions yet</td></tr>';
        return;
    }
    tbody.innerHTML = preds.map(p => `
        <tr>
            <td style="font-family:var(--mono);font-size:0.75rem;color:var(--txt3)">${new Date(p.timestamp).toLocaleString()}</td>
            <td style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(p.text)}</td>
            <td><strong style="color:var(--txt)">${p.prediction}</strong></td>
            <td style="font-family:var(--mono)">${(p.confidence * 100).toFixed(1)}%</td>
            <td class="${p.risk_flag ? 'tag-crisis' : 'tag-safe'}">${p.risk_flag ? '⚠ CRISIS' : '✓ Safe'}</td>
        </tr>`).join('');
}

// ============================================================
// CHATBOT
// ============================================================

// Condition color map for assessment display
const CONDITION_STYLES = {
    'Normal':          { bg: 'rgba(47,184,138,0.1)',   color: '#2fb88a', icon: '✅', label: 'Feeling okay' },
    'Anxiety':         { bg: 'rgba(232,169,74,0.12)',  color: '#c8923a', icon: '😰', label: 'Signs of anxiety' },
    'Depression':      { bg: 'rgba(79,128,212,0.12)',  color: '#5a80c8', icon: '💙', label: 'Low mood signals' },
    'BPD':             { bg: 'rgba(124,108,212,0.12)', color: '#8b7cd4', icon: '🌊', label: 'Emotional fluctuation' },
    'Bipolar':         { bg: 'rgba(200,100,160,0.12)', color: '#c864a0', icon: '🎭', label: 'Mood variability' },
    'Mental Illness':  { bg: 'rgba(120,130,148,0.12)', color: '#7a8294', icon: '🧠', label: 'Emotional distress' },
    'Schizophrenia':   { bg: 'rgba(200,100,100,0.12)', color: '#c86464', icon: '⚡', label: 'Distorted thinking' },
    'Stress':          { bg: 'rgba(220,140,60,0.12)',  color: '#dc8c3c', icon: '😓', label: 'Elevated stress' },
    'Loneliness':      { bg: 'rgba(79,128,212,0.12)',  color: '#6090c8', icon: '💙', label: 'Feeling isolated' },
    'Crisis':          { bg: 'rgba(200,80,80,0.2)',    color: '#c85050', icon: '🚨', label: 'Needs attention' },
    'Greeting':        { bg: 'rgba(47,184,138,0.08)',  color: '#5a6478', icon: '👋', label: 'Greeting' },
    'Gratitude':       { bg: 'rgba(47,184,138,0.08)',  color: '#5a6478', icon: '🙏', label: 'Gratitude' },
    'SmallTalk':       { bg: 'rgba(79,128,212,0.08)',  color: '#5a6478', icon: '💬', label: 'Conversation' },
    'Unknown':         { bg: 'rgba(90,100,120,0.08)',  color: '#5a6478', icon: '💭', label: 'Still listening…' },
};

function buildAssessmentHTML(assessment) {
    if (!assessment) return '';

    const cond = assessment.detected_condition || 'Unknown';
    const style = CONDITION_STYLES[cond] || CONDITION_STYLES['Unknown'];
    const prob = assessment.confidence != null ? (assessment.confidence * 100).toFixed(0) : null;
    const isNormal = assessment.is_normal;

    // Bar color based on probability
    const barColor = prob >= 70 ? '#f87171' : prob >= 40 ? '#fbbf24' : '#60a5fa';

    let html = `<div class="assessment-card" style="
        background: ${style.bg};
        border: 1px solid ${style.color}33;
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 10px;
        font-size: 0.82rem;
    ">`;

    // Top row: pill + status
    html += `<div style="display:flex; align-items:center; gap:10px; margin-bottom: ${prob != null ? '10px' : '0'};">`;    
    html += `<span style="
        background: ${style.bg};
        color: ${style.color};
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.78rem;
        white-space: nowrap;
        border: 1px solid ${style.color}44;
    ">${style.icon} ${style.label}</span>`;

    if (isNormal) {
        html += `<span style="color:#34d399;font-size:0.75rem;font-weight:500;margin-left:auto;">● No concerns detected</span>`;
    } else if (cond === 'Crisis') {
        html += `<span style="color:#ef4444;font-size:0.75rem;font-weight:600;margin-left:auto;">⚠ Seek immediate support</span>`;
    } else if (prob != null) {
        html += `<span style="color:var(--txt3);font-size:0.74rem;margin-left:auto;font-family:var(--mono);">${prob}% probability</span>`;
    }
    html += `</div>`;

    // Probability bar
    if (prob != null && cond !== 'Greeting' && cond !== 'Gratitude' && cond !== 'SmallTalk') {
        html += `<div style="margin-top:2px;">
            <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:var(--txt3);margin-bottom:4px;">
                <span>Detection Probability</span>
                <span style="font-family:var(--mono);color:${barColor};font-weight:600;">${prob}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.06);border-radius:20px;height:6px;overflow:hidden;">
                <div style="width:${prob}%;background:${barColor};height:100%;border-radius:20px;transition:width 0.6s ease;"></div>
            </div>
        </div>`;
    }

    html += `</div>`;
    return html;
}

async function sendChat() {
    const input   = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;
    addMsg(message, 'user');
    input.value = '';

    // Show typing indicator
    const typingId = addTypingIndicator();

    try {
        const res  = await fetch(`${API_BASE}/api/chatbot`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ message }),
        });
        const data = await res.json();

        // Remove typing indicator
        removeTypingIndicator(typingId);

        // Add bot response with assessment
        addMsg(data.response, 'bot', data.mental_health_assessment);

        if (data.suggestions?.length) {
            addMsg('💡 ' + data.suggestions.join('\n• '), 'bot');
        }
    } catch {
        removeTypingIndicator(typingId);
        addMsg("I'm sorry, I'm having trouble connecting. Please try again. 💙", 'bot');
    }
}

let typingCounter = 0;

function addTypingIndicator() {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    const id = `typing-${++typingCounter}`;
    div.id = id;
    div.className = 'msg bot';
    div.innerHTML = `
        <div class="msg-avatar">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"/><path d="M12 11V7"/><circle cx="12" cy="5" r="2"/></svg>
        </div>
        <div class="typing-dots"><span></span><span></span><span></span></div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function addMsg(text, sender, assessment = null) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `msg ${sender}`;

    const avatarSvg = sender === 'bot'
        ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"/><path d="M12 11V7"/><circle cx="12" cy="5" r="2"/></svg>`
        : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;

    let bubbleContent = escapeHtml(text);

    // Add assessment card if present (only for bot messages with clinical detections)
    let assessmentHTML = '';
    if (sender === 'bot' && assessment) {
        assessmentHTML = buildAssessmentHTML(assessment);
    }

    div.innerHTML = `<div class="msg-avatar">${avatarSvg}</div><div class="msg-bubble">${bubbleContent}${assessmentHTML}</div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// ============================================================
// HISTORY
// ============================================================
async function loadHistory() {
    try {
        const res  = await fetch(`${API_BASE}/api/history?limit=100`);
        const data = await res.json();
        const tbody = document.getElementById('historyTableBody');

        if (!data || data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="empty-row">No history yet</td></tr>';
            return;
        }

        tbody.innerHTML = data.map(p => `
            <tr>
                <td style="font-family:var(--mono);font-size:0.74rem;color:var(--txt3);white-space:nowrap">${new Date(p.timestamp).toLocaleString()}</td>
                <td style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(p.text)}</td>
                <td><strong style="color:var(--txt)">${softConditionName(p.prediction)}</strong></td>
                <td style="font-family:var(--mono)">${Math.round(p.confidence * 100)}%</td>
                <td class="${p.risk_flag ? 'tag-crisis' : 'tag-safe'}">${p.risk_flag ? '⚠ Needs care' : '✓ Okay'}</td>
                <td style="font-family:var(--mono);color:var(--txt3)">${p.latency_ms ? Math.round(p.latency_ms) + 'ms' : '—'}</td>
            </tr>`).join('');
    } catch {
        console.warn('History unavailable');
    }
}

// ============================================================
// PROFILE SCANNER
// ============================================================
async function runProfileScan() {
    const text = document.getElementById('batchInputText').value.trim();
    if (!text) {
        showToast('Please enter multiple posts separated by new lines.');
        return;
    }

    const lines = text.split('\\n').map(l => l.trim()).filter(l => l.length > 5);
    if (lines.length === 0) {
        showToast('Please enter valid text lines.');
        return;
    }

    const btn = document.getElementById('scanBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Scanning...';

    try {
        const res = await fetch(`${API_BASE}/api/predict/batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts: lines })
        });
        
        if (!res.ok) throw new Error('Batch scan failed');
        const data = await res.json();
        
        document.getElementById('profileResults').style.display = 'block';
        document.getElementById('scanCountLabel').textContent = `Based on ${data.count} posts`;
        
        const preds = data.predictions;
        const sumRisk = preds.reduce((a, b) => a + b.risk_score, 0);
        const avgRisk = Math.round(sumRisk / preds.length);
        document.getElementById('profRiskScore').textContent = `${avgRisk}/100`;
        
        const counts = {};
        preds.forEach(p => counts[p.prediction] = (counts[p.prediction] || 0) + 1);
        const domCond = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        document.getElementById('profDomCond').textContent = softConditionName(domCond);
        
        let summaryText;
        if (avgRisk > 60) {
            summaryText = `These messages show signs of significant emotional distress and mood variability. It may help to talk to someone you trust, or consider reaching out to a professional.`;
        } else if (avgRisk > 30) {
            summaryText = `There are some signs of emotional stress and low mood across these messages. Taking small breaks, connecting with people, and monitoring how you're feeling may help.`;
        } else {
            summaryText = `The messages show relatively stable emotional patterns. Keep taking care of yourself — even on good days, small habits make a difference.`;
        }
        document.getElementById('profSummary').textContent = summaryText;

        // Render trend chart
        const canvas = document.getElementById('profileTrendChart');
        if (profileChartInstance) profileChartInstance.destroy();

        profileChartInstance = new Chart(canvas, {
            type: 'line',
            data: {
                labels: preds.map((_, i) => `Post ${i+1}`),
                datasets: [{
                    label: 'Risk Score',
                    data: preds.map(p => p.risk_score),
                    borderColor: '#a78bfa',
                    backgroundColor: 'rgba(167,139,250,0.1)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { min: 0, max: 100 } }
            }
        });

    } catch (err) {
        showToast('Error running batch scan: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg> Scan messages`;
    }
}

// ============================================================
// COPING TOOLS (4-7-8 Breathing)
// ============================================================
let breathingStep = 0;

function startBreathing() {
    const textEl = document.getElementById('breathText');
    const circleEl = document.getElementById('breathCircle');
    document.getElementById('btnStartBreath').disabled = true;
    document.getElementById('btnStopBreath').disabled = false;
    
    breathingStep = 0;
    
    function cycle() {
        if (breathingStep === 0) {
            textEl.textContent = 'Inhale';
            circleEl.style.transform = 'scale(1.8)';
            circleEl.style.transitionDuration = '4s';
            circleEl.style.background = 'rgba(56, 189, 248, 0.4)';
            breathingStep = 1;
            breathInterval = setTimeout(cycle, 4000);
        } else if (breathingStep === 1) {
            textEl.textContent = 'Hold';
            circleEl.style.transform = 'scale(1.8)';
            circleEl.style.background = 'rgba(251, 191, 36, 0.4)';
            circleEl.style.transitionDuration = '0.5s';
            breathingStep = 2;
            breathInterval = setTimeout(cycle, 7000);
        } else {
            textEl.textContent = 'Exhale';
            circleEl.style.transform = 'scale(1)';
            circleEl.style.transitionDuration = '8s';
            circleEl.style.background = 'rgba(56, 189, 248, 0.2)';
            breathingStep = 0;
            breathInterval = setTimeout(cycle, 8000);
        }
    }
    
    cycle();
}

function stopBreathing() {
    clearTimeout(breathInterval);
    const textEl = document.getElementById('breathText');
    const circleEl = document.getElementById('breathCircle');
    
    textEl.textContent = 'Ready';
    circleEl.style.transform = 'scale(1)';
    circleEl.style.transitionDuration = '1s';
    circleEl.style.background = 'rgba(79, 128, 212, 0.1)';
    
    document.getElementById('btnStartBreath').disabled = false;
    document.getElementById('btnStopBreath').disabled = true;
}

// ============================================================
// MOOD TIPS
// ============================================================
const MOOD_TIPS = {
    music:     { title: '🎵 Calming Sounds',   body: "Put on something gentle — lo-fi, nature sounds, or your favourite calm playlist. Even 5 minutes of intentional listening can shift your mood." },
    water:     { title: '💧 Drink Some Water',  body: "Get up, pour yourself a glass of water and drink it slowly. Dehydration quietly affects your mood and energy more than most people realise." },
    walk:      { title: '🌤️ Take a Short Walk', body: "Even 5-10 minutes outside can genuinely help. You don't need a destination — just move, look around, and breathe some fresh air." },
    write:     { title: '✍️ Write It Out',       body: "Grab a notebook or open a blank note. Write without a plan — what you're feeling, what's bothering you, or just random thoughts. Getting it out of your head helps." },
    grounding: { title: '🧘 5-4-3-2-1 Grounding', body: "Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste. This brings you back to the present moment." },
    rest:      { title: '🌙 Rest for a Moment', body: "Close your eyes for 2 minutes. No phone, no noise. Just let yourself be still. Rest isn't laziness — it's part of looking after yourself." },
};

function showMoodTip(key) {
    const tip = MOOD_TIPS[key];
    if (!tip) return;
    const box = document.getElementById('moodTipBox');
    box.style.display = 'block';
    box.innerHTML = `<strong style="color:var(--txt);display:block;margin-bottom:6px;">${tip.title}</strong>${tip.body}`;
    box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================================
// SOFT CONDITION NAMES (less clinical)
// ============================================================
function softConditionName(name) {
    const map = {
        'Depression':     'Low mood signals',
        'Anxiety':        'Signs of anxiety',
        'BPD':            'Emotional fluctuation',
        'Bipolar':        'Mood variability',
        'Schizophrenia':  'Distorted thinking signals',
        'Mental Illness': 'General emotional distress',
        'Normal':         'Feeling okay',
        'Crisis':         'Needs immediate support',
    };
    return map[name] || name;
}

// ============================================================
// UTILITIES
// ============================================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text ?? '';
    return div.innerHTML;
}

function showToast(msg, type = 'info') {
    // simple console fallback for now
    console.warn(msg);
    alert(msg);
}
