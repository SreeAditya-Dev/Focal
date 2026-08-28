import React, { useState, useEffect } from "react";
import { api } from "./api/client";
import { HealthStatus } from "./api/types";
import { Header } from "./components/Header";
import { AnalyzePage } from "./pages/AnalyzePage";
import { HistoryPage } from "./pages/HistoryPage";
import { ModelInfoPage } from "./pages/ModelInfoPage";
import { Sparkles, Github, Shield } from "lucide-react";

export function App() {
  const [activeTab, setActiveTab] = useState<"analyze" | "history" | "model">("analyze");
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const h = await api.getHealth();
        setHealth(h);
      } catch (err) {
        setHealth(null);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 20000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* Background ambient mesh */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-0 left-1/4 w-[600px] h-[400px] bg-indigo-600/10 rounded-full blur-[140px]" />
        <div className="absolute top-1/3 right-1/4 w-[500px] h-[350px] bg-emerald-600/10 rounded-full blur-[140px]" />
      </div>

      {/* Header Bar */}
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        health={health}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 lg:px-8 pt-6 z-10">
        {activeTab === "analyze" && <AnalyzePage />}
        {activeTab === "history" && <HistoryPage />}
        {activeTab === "model" && <ModelInfoPage />}
      </main>

      {/* Footer */}
      <footer className="glass-panel border-t border-slate-800/80 py-6 px-4 text-center text-xs text-slate-400 z-10">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-bold text-slate-200">FOCAL</span>
            <span>— AI-Powered Image Quality & Defect Detection System</span>
          </div>
          <div className="flex items-center gap-4 text-slate-400 font-mono text-[11px]">
            <span>FastAPI + PyTorch + React</span>
            <span>•</span>
            <span>MobileNetV3 + 47-Feature Fusion</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;

