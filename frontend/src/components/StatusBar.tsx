import styles from './StatusBar.module.css';

interface StatusBarProps {
  status: 'idle' | 'online' | 'offline';
}

export default function StatusBar({ status }: StatusBarProps) {
  const labels: Record<string, string> = {
    idle: 'Connecting…',
    online: 'API Online',
    offline: 'API Offline',
  };

  return (
    <div className={`${styles.bar} ${styles[status]}`}>
      <span className={styles.dot} />
      <span className={styles.label}>{labels[status]}</span>
    </div>
  );
}
