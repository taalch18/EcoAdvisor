import React, { useState } from 'react';
import axios from 'axios';
import './index.css';

function App() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post('http://localhost:8000/api/query', {
        query: query
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError('Failed to fetch data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>EcoWise Advisor 🌿</h1>
        <p>AI-Powered ESG Investment Assistant</p>
      </header>

      <form className="search-box" onSubmit={handleSearch}>
        <input
          type="text"
          className="search-input"
          placeholder="Enter ESG query (e.g., 'Forecast for AAPL')"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" className="search-button" disabled={loading}>
          {loading ? 'Analyzing...' : 'Ask Advisor'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="result-card">
          <div className="recommendation">
            Recommendation: {result.recommendation}
          </div>

          <div className="stats">
            <div>
              <strong>Ticker:</strong> {result.ticker || "Unknown"}
            </div>
            <div>
              <strong>Growth Prediction:</strong> {result.prediction !== null ? `${result.prediction.toFixed(2)}%` : "N/A"}
            </div>
          </div>

          <div className="context">
            <strong>Analysis Context:</strong>
            <ul>
              {result.context.map((ctx, idx) => (
                <li key={idx}>{ctx.text}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
