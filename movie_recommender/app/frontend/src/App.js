import React, { useState } from 'react';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import RecommendPage from './pages/RecommendPage';
import MetricsPage from './pages/MetricsPage';
import AboutPage from './pages/AboutPage';
import './App.css';

export default function App() {
  const [page, setPage] = useState('home');

  const navigate = (p) => {
    setPage(p);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const renderPage = () => {
    switch (page) {
      case 'recommend': return <RecommendPage navigate={navigate} />;
      case 'metrics':   return <MetricsPage />;
      case 'about':     return <AboutPage />;
      default:          return <HomePage navigate={navigate} />;
    }
  };

  return (
    <div className="app-shell">
      <Navbar current={page} navigate={navigate} />
      <main className="app-main">{renderPage()}</main>
      <footer className="app-footer">
        <div className="footer-inner">
          <span className="footer-brand">🎬 MoodFlix</span>
          <span className="footer-text">
            Powered by SVM · K-Means · TMDb Dataset
          </span>
        </div>
      </footer>
    </div>
  );
}
