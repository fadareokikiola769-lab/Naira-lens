/* =========================================================
   NairaLens — Dashboard
   Pulls real output from the Flask backend: /api/dashboard/<coin>
   (price history, sentiment, LSTM/GRU forecasts, evaluation, log)
   and /api/sentiment/<coin> (live-scored news feed).
   ========================================================= */
(async function () {
  const user = await NairaAuth.requireAuthOrRedirect();
  if (!user) return;

  document.getElementById('userName').textContent = user.name;
  document.getElementById('userEmail').textContent = user.email;
  document.getElementById('avatarInit').textContent = user.name.trim()[0].toUpperCase();
  document.getElementById('greetLine').textContent =
    `Welcome back, ${user.name.split(' ')[0]}. Here's what the pipeline is showing right now.`;

  document.getElementById('signoutBtn').addEventListener('click', NairaAuth.signOut);

  const fmt = n => (n ?? 0).toLocaleString(undefined, { maximumFractionDigits: n < 10 ? 3 : 2 });
  const fmtUsd = n => (n === null || n === undefined) ? '—' : '$' + fmt(n);
  const banner = document.getElementById('statusBanner');

  function setBanner(msg, kind) {
    if (!msg) { banner.style.display = 'none'; return; }
    banner.style.display = 'block';
    banner.style.borderColor = kind === 'err' ? 'var(--red-dim)' : 'var(--line)';
    banner.style.color = kind === 'err' ? '#F3B5B8' : 'var(--text-dim)';
    banner.innerHTML = msg;
  }

  let priceChart, sentimentChart;

  function minutesAgo(iso) {
    const mins = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.round(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.round(hrs / 24)}d ago`;
  }

  async function render(coin) {
    setBanner(
      `<span class="badge-live"></span> Running the pipeline for <b>${coin}</b> — collecting price data, scoring live news, and checking the LSTM/GRU models. First request per coin can take up to a minute while a model trains.`,
      'info'
    );

    let dash, sent;
    try {
      [dash, sent] = await Promise.all([
        NairaAuth.request(`/dashboard/${coin}`),
        NairaAuth.request(`/sentiment/${coin}`),
      ]);
    } catch (e) {
      setBanner(
        e.message && e.message.includes('reach the backend')
          ? e.message
          : `Couldn't load ${coin}: ${e.message}`,
        'err'
      );
      return;
    }
    setBanner(null);
    renderDashboard(dash, sent);
  }

  function renderDashboard(dash, sent) {
    const coin = dash.coin;
    document.getElementById('chartTitle').textContent = `${coin} · Price vs. forecast (daily)`;
    document.getElementById('coinTagSent').textContent = coin;

    const prices = dash.market.prices;
    const dates = dash.market.dates;
    const last = prices[prices.length - 1];
    const prev = prices[prices.length - 2] ?? last;
    const change = prev ? ((last - prev) / prev * 100) : 0;

    const fcLstm = dash.forecast.lstm_sentiment;
    const forecastEnd = fcLstm[fcLstm.length - 1];
    const forecastChange = last ? ((forecastEnd - last) / last * 100) : 0;

    const composite = dash.sentiment.composite ?? 0;

    const stats = [
      { label: 'Live price', val: fmtUsd(last), delta: `${change >= 0 ? '+' : ''}${change.toFixed(2)}% vs prior close`, up: change >= 0 },
      { label: 'History window', val: `${prices.length} days`, delta: 'pulled from CoinGecko', up: true },
      { label: `Forecast (+${dash.forecast.horizon_labels.length}d)`, val: fmtUsd(forecastEnd), delta: `${forecastChange >= 0 ? '+' : ''}${forecastChange.toFixed(2)}% projected`, up: forecastChange >= 0 },
      { label: 'Sentiment (recent)', val: composite.toFixed(2), delta: composite >= 0 ? 'net positive' : 'net negative', up: composite >= 0 },
    ];
    document.getElementById('statRow').innerHTML = stats.map(s => `
      <div class="stat">
        <label>${s.label}</label>
        <div class="val">${s.val}</div>
        <div class="delta ${s.up ? 'up' : 'down'}">${s.delta}</div>
      </div>`).join('');

    /* ---- price + forecast chart ---- */
    const allLabels = [...dates, ...dash.forecast.horizon_labels];
    const historical = [...prices, ...Array(dash.forecast.horizon_labels.length).fill(null)];
    const bridge = prices[prices.length - 1];
    const modelForecast = [...Array(prices.length - 1).fill(null), bridge, ...dash.forecast.lstm_sentiment];
    const baseForecast = [...Array(prices.length - 1).fill(null), bridge, ...dash.forecast.baseline];

    if (priceChart) priceChart.destroy();
    priceChart = new Chart(document.getElementById('priceChart'), {
      type: 'line',
      data: {
        labels: allLabels, datasets: [
          { label: 'Historical', data: historical, borderColor: '#7C8CE0', backgroundColor: 'rgba(124,140,224,.08)', fill: true, tension: .3, pointRadius: 0, borderWidth: 2 },
          { label: 'LSTM + sentiment', data: modelForecast, borderColor: '#14B87F', borderDash: [5, 4], tension: .3, pointRadius: 0, borderWidth: 2 },
          { label: 'Baseline', data: baseForecast, borderColor: '#647389', borderDash: [2, 3], tension: .3, pointRadius: 0, borderWidth: 1.5 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false }, tooltip: {
            backgroundColor: '#121D30', borderColor: '#223349', borderWidth: 1, titleColor: '#EAF1F8', bodyColor: '#9DB0C4'
          }
        },
        scales: {
          x: { grid: { color: '#1A2841' }, ticks: { color: '#647389', maxTicksLimit: 10, font: { family: 'JetBrains Mono', size: 10 } } },
          y: { grid: { color: '#1A2841' }, ticks: { color: '#647389', font: { family: 'JetBrains Mono', size: 10 }, callback: v => fmtUsd(v) } }
        }
      }
    });

    /* ---- sentiment chart ---- */
    const sentDates = dash.sentiment.dates;
    const sentScores = dash.sentiment.scores;
    if (sentimentChart) sentimentChart.destroy();
    sentimentChart = new Chart(document.getElementById('sentimentChart'), {
      type: 'bar',
      data: {
        labels: sentDates, datasets: [{
          data: sentScores,
          backgroundColor: sentScores.map(v => v >= 0 ? 'rgba(20,184,127,.65)' : 'rgba(229,89,95,.65)'),
          borderRadius: 2, barPercentage: .9, categoryPercentage: .9
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { backgroundColor: '#121D30', borderColor: '#223349', borderWidth: 1 } },
        scales: {
          x: { display: false },
          y: { min: -1, max: 1, grid: { color: '#1A2841' }, ticks: { color: '#647389', font: { family: 'JetBrains Mono', size: 10 } } }
        }
      }
    });

    /* ---- gauge ---- */
    const pct = ((composite + 1) / 2 * 100).toFixed(1);
    document.getElementById('gaugeNum').textContent = (composite >= 0 ? '+' : '') + composite.toFixed(2);
    document.getElementById('gaugeFill').style.width = '100%';
    document.getElementById('gaugeMarker').style.left = `calc(${pct}% - 1px)`;
    document.getElementById('vaderScore').textContent = dash.sentiment.vader_mean.toFixed(2) + ' compound';
    document.getElementById('robertaScore').textContent = dash.sentiment.roberta_mean.toFixed(2) + ' compound';

    /* ---- model evaluation ---- */
    const ev = dash.evaluation;
    document.getElementById('m-lstm-sent').textContent = `RMSE ${fmt(ev.lstm_sentiment.rmse)} · MAPE ${ev.lstm_sentiment.mape}%`;
    document.getElementById('m-gru-sent').textContent = `RMSE ${fmt(ev.gru_sentiment.rmse)} · MAPE ${ev.gru_sentiment.mape}%`;
    document.getElementById('m-lstm-base').textContent = `RMSE ${fmt(ev.baseline.rmse)} · MAPE ${ev.baseline.mape}%`;

    /* ---- feed ---- */
    const feed = sent.feed || [];
    document.getElementById('feed').innerHTML = feed.length ? feed.map(item => `
      <div class="feed-item">
        <div class="meta"><span>${item.source} · ${minutesAgo(item.timestamp)}</span><span class="pill ${item.label === 'positive' ? 'pos' : item.label === 'negative' ? 'neg' : 'neu'}">${item.label}</span></div>
        <div class="txt">${item.text}</div>
      </div>`).join('') : `<p class="field-note">No recent headlines matched ${coin} — showing general market coverage next refresh.</p>`;

    /* ---- forecast log ---- */
    const rows = dash.log || [];
    document.getElementById('logBody').innerHTML = rows.length ? rows.map(r => `
      <tr>
        <td>${r.timestamp}</td>
        <td>${r.coin}</td>
        <td>${r.model}</td>
        <td>${fmtUsd(r.predicted)}</td>
        <td>${fmtUsd(r.actual)}</td>
        <td>${r.mape === null || r.mape === undefined ? '—' : r.mape.toFixed(2) + '%'}</td>
      </tr>`).join('') : `<tr><td colspan="6">No forecast history yet for this coin.</td></tr>`;
  }

  document.getElementById('coinSelect').addEventListener('click', function (e) {
    const btn = e.target.closest('button[data-coin]');
    if (!btn) return;
    [...this.querySelectorAll('button')].forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    render(btn.dataset.coin);
  });

  render('BTC');
})();
