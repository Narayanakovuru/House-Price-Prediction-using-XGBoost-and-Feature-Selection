import styles from './PriceResult.module.css';

interface PriceResultProps {
  price: number;
  onReset: () => void;
}

function formatUSD(n: number) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(n);
}

function getPriceBand(price: number): { label: string; color: string } {
  if (price < 100_000) return { label: 'Entry Level', color: '#06b6d4' };
  if (price < 175_000) return { label: 'Affordable', color: '#10b981' };
  if (price < 250_000) return { label: 'Mid Range', color: '#6366f1' };
  if (price < 400_000) return { label: 'Premium', color: '#f59e0b' };
  return { label: 'Luxury', color: '#f43f5e' };
}

export default function PriceResult({ price, onReset }: PriceResultProps) {
  const band = getPriceBand(price);
  const low = Math.round(price * 0.92);
  const high = Math.round(price * 1.08);

  return (
    <div className={styles.overlay}>
      <div className={styles.card}>
        {/* Glow orb */}
        <div className={styles.orb} style={{ background: `radial-gradient(circle, ${band.color}33 0%, transparent 70%)` }} />

        <div className={styles.tag} style={{ color: band.color, borderColor: band.color + '44', background: band.color + '11' }}>
          {band.label}
        </div>

        <p className={styles.subtitle}>Estimated Sale Price</p>

        <div className={styles.price} style={{ color: band.color }}>
          {formatUSD(price)}
        </div>

        <p className={styles.range}>
          Confidence range: <strong>{formatUSD(low)}</strong> — <strong>{formatUSD(high)}</strong>
          <span className={styles.confidence}>± 8% (R² = 0.91)</span>
        </p>

        <div className={styles.stats}>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Model</span>
            <span className={styles.statValue}>XGBoost</span>
          </div>
          <div className={styles.divider} />
          <div className={styles.stat}>
            <span className={styles.statLabel}>Accuracy</span>
            <span className={styles.statValue}>R² 0.9108</span>
          </div>
          <div className={styles.divider} />
          <div className={styles.stat}>
            <span className={styles.statLabel}>RMSE</span>
            <span className={styles.statValue}>0.129</span>
          </div>
        </div>

        <button className={styles.resetBtn} onClick={onReset}>
          ← Predict Another
        </button>
      </div>
    </div>
  );
}
