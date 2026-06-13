const API_BASE = 'http://127.0.0.1:8000';

export interface PredictRequest {
  data: Record<string, number | string>;
}

export interface PredictResponse {
  predicted_price: number;
}

export interface HealthResponse {
  status: string;
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('API unreachable');
  return res.json();
}

export async function predictPrice(data: Record<string, number | string>): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}
