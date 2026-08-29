import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { AnalyzePage } from './pages/AnalyzePage';
import { HistoryPage } from './pages/HistoryPage';
import { ModelInfoPage } from './pages/ModelInfoPage';
import { checkHealth } from './api/client';
import { HealthResponse } from './api/types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'analyze' | 'history' | 'model'>('analyze');
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await checkHealth();
        setHealth(res);
      } catch (err) {
        console.error('Health check failed', err);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} health={health} />
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'analyze' && <AnalyzePage />}
        {activeTab === 'history' && <HistoryPage />}
        {activeTab === 'model' && <ModelInfoPage />}
      </main>
      <footer className="border-t border-border py-6 text-center text-xs text-muted-foreground font-medium">
        Focal AI Vision Quality Intelligence • Powered by FastAPI & MobileNetV3
      </footer>
    </div>
  );
};

export default App;

