import { useState, useCallback } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || '';

export function useRecommend() {
  const [movies,   setMovies]   = useState([]);
  const [label,    setLabel]    = useState('');
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);

  const fetchRecs = useCallback(async (mood, model = 'svm', topN = 5) => {
    setLoading(true);
    setError(null);
    setMovies([]);

    try {
      const res = await fetch(`${API_BASE}/api/recommend`, {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify({ mood, model, top_n: topN }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Server error');
      }

      const data = await res.json();
      setMovies(data.movies);
      setLabel(data.predicted_label);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return { movies, label, loading, error, fetchRecs };
}

export function useMetrics() {
  const [metrics,  setMetrics]  = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);

  const fetchMetrics = useCallback(async () => {
    setLoading(true);
    try {
      const res  = await fetch(`${API_BASE}/api/metrics`);
      const data = await res.json();
      setMetrics(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return { metrics, loading, error, fetchMetrics };
}
