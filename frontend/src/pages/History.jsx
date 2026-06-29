import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { getHistory, getPredictionPdf } from '../services/api';
import GlassCard from '../components/UI/GlassCard';
import RiskBadge from '../components/UI/RiskBadge';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import NeonButton from '../components/UI/NeonButton';

const DISEASE_MAP = {
  diabetes: { icon: '🩸', label: 'Diabetes' },
  heart: { icon: '❤️', label: 'Heart Disease' },
  parkinsons: { icon: '🧠', label: "Parkinson's" },
};

export default function History() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [downloading, setDownloading] = useState(null);

  const fetchHistory = () => {
    setLoading(true);
    getHistory()
      .then(res => setPredictions(res.data.predictions || []))
      .catch(() => setError('Failed to load history'))
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchHistory(); }, []);

  const handleDownload = async (id) => {
    setDownloading(id);
    try {
      const res = await getPredictionPdf(id);
      const url = URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = `prediction_report_${id}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError('Failed to download PDF');
    } finally {
      setDownloading(null);
    }
  };

  const getFilterBadge = (prediction) => {
    const severity = prediction.confidence >= 0.80 ? 'high' : prediction.confidence >= 0.40 ? 'moderate' : 'low';
    return <RiskBadge level={severity} />;
  };

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Prediction History</h1>
          <p className="text-slate-400">View and download your past medical assessments</p>
        </div>
        <NeonButton variant="outline" onClick={fetchHistory} className="text-sm">
          Refresh
        </NeonButton>
      </motion.div>

      {error && (
        <div className="p-3 rounded-lg bg-danger-500/10 border border-danger-500/20 text-danger-400 text-sm">{error}</div>
      )}

      {loading ? (
        <LoadingSpinner className="py-20" />
      ) : predictions.length === 0 ? (
        <GlassCard className="text-center py-16">
          <div className="text-5xl mb-4">📋</div>
          <h3 className="text-lg font-semibold text-white mb-2">No Predictions Yet</h3>
          <p className="text-slate-400 text-sm mb-6">Your diagnosis history will appear here once you make your first prediction.</p>
          <a href="/predict" className="neon-btn inline-block text-sm">Start Diagnosis</a>
        </GlassCard>
      ) : (
        <div className="space-y-3">
          {predictions.map((p, i) => {
            const disease = DISEASE_MAP[p.disease_type] || { icon: '🔬', label: p.disease_type };
            const confidencePct = (p.confidence * 100).toFixed(1);

            return (
              <motion.div
                key={p.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.03 }}
              >
                <GlassCard hover={false} className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-xl flex-shrink-0">
                    {disease.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <h3 className="text-sm font-semibold text-white capitalize">{disease.label}</h3>
                      {getFilterBadge(p)}
                    </div>
                    <p className="text-xs text-slate-500">
                      {new Date(p.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                      <span className="mx-2">·</span>
                      Confidence: {confidencePct}%
                    </p>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className="text-sm font-bold text-white">{p.prediction_result === 1 ? '⚠️' : '✅'}</span>
                    <NeonButton
                      variant="outline"
                      className="text-xs !py-1.5 !px-3"
                      loading={downloading === p.id}
                      onClick={() => handleDownload(p.id)}
                    >
                      PDF
                    </NeonButton>
                  </div>
                </GlassCard>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}
