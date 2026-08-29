import React from 'react';
import { Eye, Activity, History, Info, Sparkles } from 'lucide-react';
import { HealthResponse } from '../api/types';
import { ThemeToggle } from './ThemeToggle';

interface HeaderProps {
  activeTab: 'analyze' | 'history' | 'model';
  setActiveTab: (tab: 'analyze' | 'history' | 'model') => void;
  health: HealthResponse | null;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, health }) => {
  return (
    <header className="border-b border-border bg-background/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-primary to-primary/70 flex items-center justify-center shadow-lg shadow-primary/25">
            <Eye className="w-5 h-5 text-primary-foreground" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-xl tracking-tight text-foreground">
                FOCAL
              </span>
              <span className="text-[10px] uppercase font-bold tracking-widest px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20">
                AI Vision
              </span>
            </div>
            <p className="text-xs text-muted-foreground font-medium">Image Quality & Defect Intelligence</p>
          </div>
        </div>

        <nav className="flex items-center gap-1 bg-secondary/50 p-1 rounded-xl border border-border">
          <button
            onClick={() => setActiveTab('analyze')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
              activeTab === 'analyze'
                ? 'bg-primary text-primary-foreground shadow-md shadow-primary/30'
                : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
            }`}
          >
            <Sparkles className="w-4 h-4" />
            Analyze
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
              activeTab === 'history'
                ? 'bg-primary text-primary-foreground shadow-md shadow-primary/30'
                : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
            }`}
          >
            <History className="w-4 h-4" />
            History
          </button>
          <button
            onClick={() => setActiveTab('model')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
              activeTab === 'model'
                ? 'bg-primary text-primary-foreground shadow-md shadow-primary/30'
                : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
            }`}
          >
            <Info className="w-4 h-4" />
            Architecture
          </button>
        </nav>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-secondary/40 border border-border text-xs">
            <Activity
              className={`w-3.5 h-3.5 ${
                health?.status === 'ok' ? 'text-emerald-500 animate-pulse' : 'text-amber-500'
              }`}
            />
            <span className="text-muted-foreground">Model:</span>
            <span className="font-mono font-medium text-foreground">
              {health?.model_version || 'Loading...'}
            </span>
          </div>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
};

