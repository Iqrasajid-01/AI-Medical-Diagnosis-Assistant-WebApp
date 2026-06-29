import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { predictDiabetes, predictHeart, predictParkinsons } from '../services/api';
import { recordWav } from '../utils/audioRecorder';
import GlassCard from '../components/UI/GlassCard';
import NeonButton from '../components/UI/NeonButton';
import ConfidenceBar from '../components/UI/ConfidenceBar';
import RiskBadge from '../components/UI/RiskBadge';
import DisclaimerBanner from '../components/UI/DisclaimerBanner';
import DoctorRecommendation from '../components/UI/DoctorRecommendation';

const DISEASE_TABS = [
  { key: 'diabetes', label: 'Diabetes', icon: '🩸' },
  { key: 'heart', label: 'Heart Disease', icon: '❤️' },
  { key: 'parkinsons', label: "Parkinson's", icon: '🧠' },
];

function InfoIcon({ tooltip }) {
  return (
    <div className="relative group">
      <span className="text-slate-500 hover:text-primary-400 cursor-help transition-colors text-xs">ℹ️</span>
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-60 p-2.5 rounded-lg bg-dark-800 border border-white/10 text-xs text-slate-300 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50 shadow-xl">
        {tooltip}
      </div>
    </div>
  );
}

function Field({ label, name, value, onChange, type = 'number', required = false, placeholder, tooltip, optional }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-sm font-medium text-slate-400">
          {label}
          {required && <span className="text-danger-400 ml-0.5">*</span>}
          {optional && <span className="text-xs text-slate-500 ml-2 italic">(Optional)</span>}
        </label>
        {tooltip && <InfoIcon tooltip={tooltip} />}
      </div>
      <input
        type={type}
        name={name}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        step={type === 'number' ? 'any' : undefined}
        className="input-glow"
      />
    </div>
  );
}

function SelectField({ label, name, value, onChange, options, required, tooltip }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-sm font-medium text-slate-400">
          {label}
          {required && <span className="text-danger-400 ml-0.5">*</span>}
        </label>
        {tooltip && <InfoIcon tooltip={tooltip} />}
      </div>
      <select name={name} value={value} onChange={onChange} required={required} className="input-glow">
        <option value="">Select...</option>
        {options.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
}

function ResultCard({ result, onReset, onFindDoctors }) {
  const riskLevel = result.risk_level?.toLowerCase() || 'low';
  const confidencePct = (result.confidence * 100).toFixed(1);

  const titleMap = { low: 'Low Risk', moderate: 'Moderate Risk', high: 'High Risk Detected' };
  const emojiMap = { low: '✅', moderate: '⚠️', high: '⚠️' };
  const colorMap = { low: 'text-success-400', moderate: 'text-yellow-400', high: 'text-danger-400' };
  const borderMap = { low: 'border-success-500/30', moderate: 'border-yellow-500/30', high: 'border-danger-500/30' };
  const barColorMap = { low: 'success', moderate: 'warning', high: 'danger' };
  const isAlert = riskLevel === 'high' || riskLevel === 'moderate';

  return (
    <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
      <GlassCard className={`border ${borderMap[riskLevel]}`}>
        <div className="text-center mb-6">
          <div className={`text-5xl mb-3 ${isAlert ? 'animate-pulse-glow' : ''}`}>
            {emojiMap[riskLevel]}
          </div>
          <h3 className={`text-2xl font-bold ${colorMap[riskLevel]}`}>
            {titleMap[riskLevel]}
          </h3>
          <p className="text-slate-400 text-sm mt-1 capitalize">{result.disease} Assessment</p>
        </div>

        <div className="space-y-4 mb-6">
          <ConfidenceBar value={parseFloat(confidencePct)} label="Confidence Score" color={barColorMap[riskLevel]} />
          <div className="flex justify-center">
            <RiskBadge level={riskLevel} />
          </div>
        </div>

        {(riskLevel === 'high' || riskLevel === 'moderate') && (
          <div className="mb-4">
            <button
              onClick={onFindDoctors}
              className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-500 hover:to-primary-400 text-white font-medium transition-all flex items-center justify-center gap-2 shadow-lg shadow-primary-500/25"
            >
              <span className="text-xl">🏥</span>
              Find Nearby Doctors
            </button>
          </div>
        )}

        <DisclaimerBanner className="mb-4" />

        <NeonButton onClick={onReset} variant="outline" className="w-full">
          New Prediction
        </NeonButton>
      </GlassCard>
    </motion.div>
  );
}

function DiabetesForm({ onResult }) {
  const [form, setForm] = useState({
    gender: '', age: '', hypertension: '', heart_disease: '', smoking_history: '',
    bmi: '', HbA1c_level: '', blood_glucose_level: '',
  });
  const [weight, setWeight] = useState('');
  const [height, setHeight] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const computeBMI = useCallback((w, h) => {
    if (w && h) {
      const hM = parseFloat(h) / 100;
      return (parseFloat(w) / (hM * hM)).toFixed(1);
    }
    return '';
  }, []);

  const handleWeightChange = (e) => {
    const w = e.target.value;
    setWeight(w);
    const bmi = computeBMI(w, height);
    if (bmi) setForm(prev => ({ ...prev, bmi }));
  };

  const handleHeightChange = (e) => {
    const h = e.target.value;
    setHeight(h);
    const bmi = computeBMI(weight, h);
    if (bmi) setForm(prev => ({ ...prev, bmi }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const payload = {};
      const required = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'];
      for (const key of required) {
        if (!form[key] && form[key] !== 0) throw new Error(`${key} is required`);
        payload[key] = parseFloat(form[key]);
      }
      if (form.gender) payload.gender = form.gender;
      if (form.hypertension) payload.hypertension = parseFloat(form.hypertension);
      if (form.heart_disease) payload.heart_disease = parseFloat(form.heart_disease);
      if (form.smoking_history) payload.smoking_history = form.smoking_history;

      const res = await predictDiabetes(payload);
      onResult(res.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div className="grid sm:grid-cols-2 gap-4">
        <Field label="Age" name="age" value={form.age} onChange={handleChange} required placeholder="e.g. 45" />
        <SelectField label="Gender" name="gender" value={form.gender} onChange={handleChange}
          options={[
            { value: 'Male', label: 'Male' },
            { value: 'Female', label: 'Female' },
          ]}
          tooltip="Demographic information for risk calculation." />
      </div>
      <div className="grid sm:grid-cols-2 gap-4">
        <Field label="Weight (kg)" name="weight" value={weight} onChange={handleWeightChange} placeholder="e.g. 70" />
        <Field label="Height (cm)" name="height" value={height} onChange={handleHeightChange} placeholder="e.g. 175" />
      </div>
      <Field label="BMI (auto-calculated)" name="bmi" value={form.bmi} onChange={handleChange} required placeholder="Or enter manually" />
      <div className="grid sm:grid-cols-2 gap-4">
        <Field label="HbA1c Level" name="HbA1c_level" value={form.HbA1c_level} onChange={handleChange} required placeholder="e.g. 5.5" tooltip="Glycated hemoglobin (%). Key diabetes marker." />
        <Field label="Blood Glucose" name="blood_glucose_level" value={form.blood_glucose_level} onChange={handleChange} required placeholder="e.g. 140" tooltip="Blood glucose level (mg/dL)." />
      </div>

      <button type="button" onClick={() => document.getElementById('diabetes-advanced').classList.toggle('hidden')} className="flex items-center gap-2 text-sm text-primary-400 hover:text-primary-300 transition-colors">
        ▶ Advanced Fields
        <span className="text-xs text-slate-500">(optional)</span>
      </button>

      <div id="diabetes-advanced" className="hidden space-y-4">
        <div className="grid sm:grid-cols-2 gap-4">
          <SelectField label="Hypertension" name="hypertension" value={form.hypertension} onChange={handleChange}
            options={[
              { value: '1', label: 'Yes' },
              { value: '0', label: 'No' },
            ]} />
          <SelectField label="Heart Disease" name="heart_disease" value={form.heart_disease} onChange={handleChange}
            options={[
              { value: '1', label: 'Yes' },
              { value: '0', label: 'No' },
            ]} />
        </div>
        <SelectField label="Smoking History" name="smoking_history" value={form.smoking_history} onChange={handleChange}
          options={[
            { value: 'never', label: 'Never smoked' },
            { value: 'former', label: 'Former smoker' },
            { value: 'current', label: 'Current smoker' },
            { value: 'not current', label: 'Not currently smoking' },
            { value: 'ever', label: 'Ever smoked' },
            { value: 'No Info', label: 'No information' },
          ]} />
      </div>

      {error && <p className="text-danger-400 text-sm">{error}</p>}

      <NeonButton type="submit" loading={loading} className="w-full">
        Predict Diabetes Risk
      </NeonButton>
    </form>
  );
}

function HeartForm({ onResult }) {
  const [form, setForm] = useState({
    Age: '', Sex: '', ChestPainType: '', RestingBP: '', Cholesterol: '',
    FastingBS: '', RestingECG: '', MaxHR: '', ExerciseAngina: '', Oldpeak: '', ST_Slope: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const payload = {};
      const required = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'MaxHR'];
      for (const key of required) {
        if (!form[key] && form[key] !== 0) throw new Error(`${key} is required`);
        payload[key] = parseFloat(form[key]) || form[key];
      }
      const requiredStr = ['Sex', 'ChestPainType'];
      for (const key of requiredStr) {
        payload[key] = form[key];
      }
      const optional = ['Cholesterol', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'];
      for (const key of optional) {
        if (form[key] !== '' && form[key] !== undefined) {
          payload[key] = parseFloat(form[key]) || form[key];
        }
      }

      const res = await predictHeart(payload);
      onResult(res.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div className="grid sm:grid-cols-3 gap-4">
        <Field label="Age" name="Age" value={form.Age} onChange={handleChange} required placeholder="e.g. 55" />
        <SelectField label="Sex" name="Sex" value={form.Sex} onChange={handleChange} required
          options={[
            { value: 'M', label: 'Male' },
            { value: 'F', label: 'Female' },
          ]} />
        <SelectField label="Chest Pain Type" name="ChestPainType" value={form.ChestPainType} onChange={handleChange} required
          options={[
            { value: 'TA', label: 'Typical Angina' },
            { value: 'ATA', label: 'Atypical Angina' },
            { value: 'NAP', label: 'Non-Anginal Pain' },
            { value: 'ASY', label: 'Asymptomatic' },
          ]} />
      </div>
      <div className="grid sm:grid-cols-2 gap-4">
        <Field label="Resting Blood Pressure" name="RestingBP" value={form.RestingBP} onChange={handleChange} required placeholder="e.g. 130" />
        <Field label="Max Heart Rate" name="MaxHR" value={form.MaxHR} onChange={handleChange} required placeholder="e.g. 150" />
      </div>

      <button type="button" onClick={() => document.getElementById('heart-advanced').classList.toggle('hidden')} className="flex items-center gap-2 text-sm text-primary-400 hover:text-primary-300 transition-colors">
        ▶ Advanced Fields
        <span className="text-xs text-slate-500">(optional)</span>
      </button>

      <div id="heart-advanced" className="hidden space-y-4">
        <div className="grid sm:grid-cols-2 gap-4">
          <Field label="Cholesterol" name="Cholesterol" value={form.Cholesterol} onChange={handleChange} placeholder="e.g. 240" tooltip="Serum cholesterol (mg/dl)." />
          <SelectField label="Fasting Blood Sugar &gt; 120" name="FastingBS" value={form.FastingBS} onChange={handleChange}
            options={[
              { value: '1', label: 'True' },
              { value: '0', label: 'False' },
            ]} />
        </div>
        <div className="grid sm:grid-cols-2 gap-4">
          <SelectField label="Resting ECG" name="RestingECG" value={form.RestingECG} onChange={handleChange}
            options={[
              { value: 'Normal', label: 'Normal' },
              { value: 'ST', label: 'ST-T Wave Abnormality' },
              { value: 'LVH', label: 'Left Ventricular Hypertrophy' },
            ]} />
          <SelectField label="Exercise Angina" name="ExerciseAngina" value={form.ExerciseAngina} onChange={handleChange}
            options={[
              { value: 'Y', label: 'Yes' },
              { value: 'N', label: 'No' },
            ]} />
        </div>
        <div className="grid sm:grid-cols-2 gap-4">
          <Field label="Oldpeak (ST Depression)" name="Oldpeak" value={form.Oldpeak} onChange={handleChange} placeholder="e.g. 1.5" />
          <SelectField label="ST Slope" name="ST_Slope" value={form.ST_Slope} onChange={handleChange}
            options={[
              { value: 'Up', label: 'Upsloping' },
              { value: 'Flat', label: 'Flat' },
              { value: 'Down', label: 'Downsloping' },
            ]} />
        </div>
      </div>

      {error && <p className="text-danger-400 text-sm">{error}</p>}

      <NeonButton type="submit" loading={loading} className="w-full">
        Predict Heart Disease Risk
      </NeonButton>
    </form>
  );
}

function ParkinsonsForm({ onResult }) {
  const [file, setFile] = useState(null);
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const abortRef = useRef(null);

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (f) {
      if (!f.name.endsWith('.wav')) {
        setError('Only .wav files are accepted');
        return;
      }
      if (f.size > 2 * 1024 * 1024) {
        setError('File too large (max 2MB)');
        return;
      }
      setFile(f);
      setAudioUrl(URL.createObjectURL(f));
      setError('');
    }
  };

  const stopRecording = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  };

  const startRecording = async () => {
    try {
      setRecording(true);
      setError('');
      const controller = new AbortController();
      abortRef.current = controller;
      const f = await recordWav(controller.signal);
      abortRef.current = null;
      setFile(f);
      setAudioUrl(URL.createObjectURL(f));
      setRecording(false);
    } catch (err) {
      setRecording(false);
      setError(err.message || 'Recording failed');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) { setError('Please upload or record an audio file'); return; }
    setError('');
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('audio', file);
      const res = await predictParkinsons(formData);
      onResult(res.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="text-center p-8 rounded-xl border-2 border-dashed border-white/10 hover:border-primary-500/30 transition-colors">
        {!audioUrl ? (
          <div className="space-y-4">
            <div className="text-5xl">🎤</div>
            <p className="text-slate-400 text-sm">Record a sustained "aaah" vowel sound (auto-stops after 8 seconds)</p>
            <div className="flex justify-center gap-3">
              {!recording ? (
                <NeonButton type="button" onClick={startRecording}>Start Recording</NeonButton>
              ) : (
                <NeonButton type="button" variant="danger" onClick={stopRecording}>Stop Recording</NeonButton>
              )}
            </div>
            <div className="relative">
              <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-white/5" /></div>
              <div className="relative flex justify-center"><span className="bg-dark-950 px-3 text-xs text-slate-600">or</span></div>
            </div>
            <label className="neon-btn-outline inline-block cursor-pointer text-sm">
              Upload .wav File
              <input type="file" accept=".wav" onChange={handleFile} className="hidden" />
            </label>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="text-5xl">✅</div>
            <p className="text-sm text-slate-300">{file?.name || 'Audio recorded'}</p>
            <NeonButton type="button" variant="outline" onClick={() => { setFile(null); setAudioUrl(null); }}>
              Remove & Re-record
            </NeonButton>
          </div>
        )}
      </div>

      <div className="p-3 rounded-lg bg-primary-500/10 border border-primary-500/20">
        <p className="text-xs text-slate-400">
          Speak clearly into your microphone. Say "aaah" for at least 3 seconds for accurate analysis.
          Your recording is processed locally and not stored permanently.
        </p>
      </div>

      {recording && (
        <div className="flex items-center gap-2 text-danger-400 text-sm">
          <span className="w-2 h-2 rounded-full bg-danger-400 animate-pulse" />
          Recording... Speak now
        </div>
      )}

      {error && <p className="text-danger-400 text-sm">{error}</p>}

      <NeonButton type="submit" loading={loading} className="w-full" disabled={!file}>
        Analyze Voice for Parkinson's
      </NeonButton>
    </form>
  );
}

export default function Prediction() {
  const [activeTab, setActiveTab] = useState('diabetes');
  const [result, setResult] = useState(null);
  const [showDoctors, setShowDoctors] = useState(false);

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-3xl font-bold text-white mb-2">AI Diagnosis</h1>
        <p className="text-slate-400">Select a disease and enter your medical parameters</p>
      </motion.div>

      <div className="flex gap-2 p-1 rounded-xl bg-white/5 border border-white/5 w-fit">
        {DISEASE_TABS.map(tab => (
          <button
            key={tab.key}
            onClick={() => { setActiveTab(tab.key); setResult(null); }}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === tab.key ? 'bg-primary-500/20 text-primary-400 shadow-sm' : 'text-slate-400 hover:text-white'
            }`}
          >
            <span>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      <DisclaimerBanner />

      <div className="grid lg:grid-cols-5 gap-6">
        <div className="lg:col-span-3">
          <GlassCard>
            <h2 className="text-lg font-semibold text-white mb-4 capitalize">{activeTab} Risk Assessment</h2>
            {result ? (
              <ResultCard
                result={result}
                onReset={() => setResult(null)}
                onFindDoctors={() => setShowDoctors(true)}
              />
            ) : (
              <>
                {activeTab === 'diabetes' && <DiabetesForm onResult={setResult} />}
                {activeTab === 'heart' && <HeartForm onResult={setResult} />}
                {activeTab === 'parkinsons' && <ParkinsonsForm onResult={setResult} />}
              </>
            )}
          </GlassCard>
        </div>

        <div className="lg:col-span-2 space-y-4">
          <GlassCard>
            <h3 className="text-sm font-semibold text-white mb-3">About This Assessment</h3>
            {activeTab === 'diabetes' && (
              <div className="text-xs text-slate-400 space-y-2">
                <p>ANN model trained on 100K patient records analyzing HbA1c, blood glucose, BMI, and demographics.</p>
                <p className="text-slate-500">Required: Age, BMI, HbA1c, Blood Glucose</p>
                <p className="text-yellow-400/80">Gender, hypertension, heart disease history, and smoking status are optional.</p>
              </div>
            )}
            {activeTab === 'heart' && (
              <div className="text-xs text-slate-400 space-y-2">
                <p>Multi-parameter cardiac risk assessment using clinical and ECG parameters from 918 patients.</p>
                <p className="text-slate-500">Required: Age, Sex, Chest Pain Type, Blood Pressure, Max Heart Rate</p>
                <p className="text-yellow-400/80">Cholesterol, ECG, angina, and ST slope are optional.</p>
              </div>
            )}
            {activeTab === 'parkinsons' && (
              <div className="text-xs text-slate-400 space-y-2">
                <p>Voice-based acoustic analysis extracts 22 features from your recording to detect Parkinsonian patterns using ANN.</p>
                <p className="text-slate-500">Record a sustained "aaah" vowel sound (3-10 seconds).</p>
                <p className="text-yellow-400/80">Use a quiet environment for best results. Files are limited to 2MB and 10 seconds.</p>
              </div>
            )}
          </GlassCard>

          <GlassCard>
            <h3 className="text-sm font-semibold text-white mb-3">Understanding Results</h3>
            <div className="space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <RiskBadge level="low" />
                <span className="text-slate-400">Low risk (&lt;40% confidence)</span>
              </div>
              <div className="flex items-center gap-2">
                <RiskBadge level="moderate" />
                <span className="text-slate-400">Moderate risk (40-79%)</span>
              </div>
              <div className="flex items-center gap-2">
                <RiskBadge level="high" />
                <span className="text-slate-400">High risk (&ge;80% confidence)</span>
              </div>
            </div>
          </GlassCard>
        </div>
      </div>

      {showDoctors && result && (
        <DoctorRecommendation
          result={result}
          onClose={() => setShowDoctors(false)}
        />
      )}
    </div>
  );
}
