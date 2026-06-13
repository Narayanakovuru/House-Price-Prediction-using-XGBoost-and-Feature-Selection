import { FormField } from '../features';
import styles from './FieldCard.module.css';

interface FieldCardProps {
  field: FormField;
  value: number | string;
  onChange: (key: string, value: number | string) => void;
}

export default function FieldCard({ field, value, onChange }: FieldCardProps) {
  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <label className={styles.label} htmlFor={field.key}>
          {field.label}
        </label>
        {field.type !== 'select' && field.unit && (
          <span className={styles.unit}>{field.unit}</span>
        )}
      </div>

      {field.description && (
        <p className={styles.description}>{field.description}</p>
      )}

      {field.type === 'slider' && (
        <div className={styles.sliderWrap}>
          <input
            id={field.key}
            type="range"
            className={styles.slider}
            min={field.min}
            max={field.max}
            step={field.step}
            value={value as number}
            onChange={e => onChange(field.key, Number(e.target.value))}
          />
          <div className={styles.sliderRow}>
            <span className={styles.sliderMin}>{field.min}</span>
            <span className={styles.sliderValue}>{value}{field.unit ? ` ${field.unit}` : ''}</span>
            <span className={styles.sliderMax}>{field.max}</span>
          </div>
        </div>
      )}

      {field.type === 'number' && (
        <div className={styles.numberWrap}>
          <button
            className={styles.stepper}
            onClick={() => onChange(field.key, Math.max(field.min, (value as number) - field.step))}
          >−</button>
          <input
            id={field.key}
            type="number"
            className={styles.numberInput}
            min={field.min}
            max={field.max}
            step={field.step}
            value={value as number}
            onChange={e => onChange(field.key, Number(e.target.value))}
          />
          <button
            className={styles.stepper}
            onClick={() => onChange(field.key, Math.min(field.max, (value as number) + field.step))}
          >+</button>
        </div>
      )}

      {field.type === 'select' && (
        <select
          id={field.key}
          className={styles.select}
          value={value as string}
          onChange={e => onChange(field.key, e.target.value)}
        >
          {field.options.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      )}
    </div>
  );
}
