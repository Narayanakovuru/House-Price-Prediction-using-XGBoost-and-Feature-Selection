import { useState, useEffect, useCallback } from 'react';
import { FIELD_GROUPS, buildDefaultValues } from './features';
import { checkHealth, predictPrice } from './api';
import FieldCard from './components/FieldCard';
import PriceResult from './components/PriceResult';
import styles from './App.module.css';

type ApiStatus = 'idle' | 'online' | 'offline';
type PredictState = 'idle' | 'loading' | 'done' | 'error';

export default function App() {
  const [values, setValues] = useState<Record<string, number | string>>(buildDefaultValues);
  const [apiStatus, setApiStatus] = useState<ApiStatus>('idle');
  const [predictState, setPredictState] = useState<PredictState>('idle');
  const [resultPrice, setResultPrice] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState('');

  useEffect(() => {
    const ping = async () => {
      try { await checkHealth(); setApiStatus('online'); }
      catch { setApiStatus('offline'); }
    };
    ping();
    const id = setInterval(ping, 15_000);
    return () => clearInterval(id);
  }, []);

  const handleChange = useCallback((key: string, value: number | string) => {
    setValues(prev => ({ ...prev, [key]: value }));
  }, []);

  const handlePredict = async () => {
    setPredictState('loading');
    setErrorMsg('');
    try {
      const res = await predictPrice(values);
      setResultPrice(res.predicted_price);
      setPredictState('done');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Prediction failed');
      setPredictState('error');
    }
  };

  const handleReset = () => { setPredictState('idle'); setResultPrice(null); };

  // Flatten all fields into a single compact view
  const allFields = FIELD_GROUPS.flatMap(g => g.fields);

  return (
    <div className={styles.root}>
      {/* ── Header ── */}
      <header className={styles.header}>
        {/* Brand */}
        <div className={styles.brand}>
          <span className={styles.brandIcon}>🏡</span>
          <div>
            <span className={styles.brandTitle}>Ames Housing</span>
            <span className={styles.brandSub}>AI Price Predictor</span>
          </div>
        </div>

        {/* Metric chips */}
        <div className={styles.chips}>
          {[
            { label: 'R²', value: '0.9108' },
            { label: 'RMSE', value: '0.129' },
            { label: 'MAE', value: '0.087' },
            { label: 'Latency', value: '<10ms' },
          ].map(m => (
            <div key={m.label} className={styles.chip}>
              <span className={styles.chipVal}>{m.value}</span>
              <span className={styles.chipKey}>{m.label}</span>
            </div>
          ))}
        </div>

        <button
          className={`${styles.predictBtn} ${predictState === 'loading' ? styles.predictLoading : ''}`}
          onClick={handlePredict}
          disabled={predictState === 'loading' || apiStatus === 'offline'}
        >
          {predictState === 'loading' ? 'Predicting…' : '🚀 Predict Price'}
        </button>

        {/* API status */}
        <div className={`${styles.status} ${styles[`status_${apiStatus}`]}`}>
          <span className={styles.statusDot} />
          {apiStatus === 'online' ? 'API Online' : apiStatus === 'offline' ? 'API Offline' : 'Connecting…'}
        </div>
      </header>

      {/* ── Body: Single Compact Grid ── */}
      <div className={styles.body}>
        {predictState === 'error' && (
          <div className={styles.errorBanner}>⚠ {errorMsg}</div>
        )}
        <div className={styles.fieldGrid}>
          {allFields.map(field => (
            <FieldCard key={field.key} field={field} value={values[field.key]} onChange={handleChange} />
          ))}
        </div>
      </div>

      {/* Result overlay */}
      {predictState === 'done' && resultPrice !== null && (
        <PriceResult price={resultPrice} onReset={handleReset} />
      )}
    </div>
  );
}
