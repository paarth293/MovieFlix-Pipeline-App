import React from 'react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  ResponsiveContainer, Tooltip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Cell,
} from 'recharts';
import './MetricsChart.css';

const MODEL_COLORS = {
  SVM: '#6B1D3A',
  KNN: '#C4663A',
  'Logistic Regression': '#7A9E7E',
};

const METRICS_LABELS = ['accuracy', 'precision', 'recall', 'f1'];

// ── Radar Chart ──────────────────────────────────────────────────────────────
export function ModelRadarChart({ metrics }) {
  const data = METRICS_LABELS.map((m) => ({
    metric: m.charAt(0).toUpperCase() + m.slice(1),
    SVM:    +(metrics['SVM']?.[m] * 100).toFixed(2),
    KNN:    +(metrics['KNN']?.[m] * 100).toFixed(2),
    LR:     +(metrics['Logistic Regression']?.[m] * 100).toFixed(2),
  }));

  return (
    <div className="chart-card">
      <h3 className="chart-title">Model Comparison — Radar</h3>
      <p className="chart-subtitle">All metrics as percentage (%)</p>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
          <PolarGrid stroke="#E0D8CC" />
          <PolarAngleAxis dataKey="metric" tick={{ fontFamily: 'Lato', fontSize: 13, fill: '#4A454F' }} />
          <Radar name="SVM" dataKey="SVM" stroke="#6B1D3A" fill="#6B1D3A" fillOpacity={0.2} strokeWidth={2} />
          <Radar name="KNN" dataKey="KNN" stroke="#C4663A" fill="#C4663A" fillOpacity={0.15} strokeWidth={2} />
          <Radar name="LR"  dataKey="LR"  stroke="#7A9E7E" fill="#7A9E7E" fillOpacity={0.15} strokeWidth={2} />
          <Legend wrapperStyle={{ fontFamily: 'Lato', fontSize: 13 }} />
          <Tooltip
            formatter={(v) => [`${v.toFixed(2)}%`]}
            contentStyle={{ fontFamily: 'Lato', background: '#F7F2E8', border: '1px solid #D4AF37', borderRadius: 8 }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Bar Chart ────────────────────────────────────────────────────────────────
export function MetricsBarChart({ metrics }) {
  const models = Object.keys(metrics);

  const data = models.map((m) => ({
    name: m === 'Logistic Regression' ? 'Log. Reg.' : m,
    Accuracy:  +(metrics[m].accuracy  * 100).toFixed(2),
    Precision: +(metrics[m].precision * 100).toFixed(2),
    Recall:    +(metrics[m].recall    * 100).toFixed(2),
    F1:        +(metrics[m].f1        * 100).toFixed(2),
  }));

  const METRIC_COLORS = {
    Accuracy:  '#6B1D3A',
    Precision: '#C4663A',
    Recall:    '#D4AF37',
    F1:        '#7A9E7E',
  };

  return (
    <div className="chart-card">
      <h3 className="chart-title">Metrics per Model — Bar Chart</h3>
      <p className="chart-subtitle">All metrics as percentage (%)</p>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E0D8CC" />
          <XAxis dataKey="name" tick={{ fontFamily: 'Lato', fontSize: 13, fill: '#4A454F' }} />
          <YAxis domain={[85, 100]} tick={{ fontFamily: 'Lato', fontSize: 12, fill: '#4A454F' }} />
          <Tooltip
            contentStyle={{ fontFamily: 'Lato', background: '#F7F2E8', border: '1px solid #D4AF37', borderRadius: 8 }}
            formatter={(v) => [`${v.toFixed(2)}%`]}
          />
          <Legend wrapperStyle={{ fontFamily: 'Lato', fontSize: 13 }} />
          {Object.entries(METRIC_COLORS).map(([key, color]) => (
            <Bar key={key} dataKey={key} fill={color} radius={[4, 4, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Metrics Table ────────────────────────────────────────────────────────────
export function MetricsTable({ metrics }) {
  const models = Object.keys(metrics);
  const best = models.reduce((a, b) =>
    metrics[a].f1 > metrics[b].f1 ? a : b
  );

  return (
    <div className="chart-card">
      <h3 className="chart-title">Model Scorecard</h3>
      <p className="chart-subtitle">★ highlights the best-performing model</p>
      <div className="metrics-table-wrap">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m} className={m === best ? 'row--best' : ''}>
                <td className="model-name-cell">
                  {m === best && <span className="best-badge">★ Best</span>}
                  {m}
                </td>
                <td>{(metrics[m].accuracy  * 100).toFixed(2)}%</td>
                <td>{(metrics[m].precision * 100).toFixed(2)}%</td>
                <td>{(metrics[m].recall    * 100).toFixed(2)}%</td>
                <td className="f1-cell">{(metrics[m].f1 * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
