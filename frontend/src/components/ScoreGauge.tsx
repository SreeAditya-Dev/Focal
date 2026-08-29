import React from 'react';
import { ShieldCheck, AlertTriangle, XCircle, CheckCircle2 } from 'lucide-react';

interface ScoreGaugeProps {
  score: number;
  label: 'EXCELLENT' | 'ACCEPTABLE' | 'POOR' | 'UNUSABLE';
  processingTimeMs?: number;
}

export const ScoreGauge: React.FC<ScoreGaugeProps> = ({ score, label, processingTimeMs }) => {
  const radius = 64;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  const getColorConfig = () => {
    switch (label) {
      case 'EXCELLENT':
        return {
          stroke: '#10b981',
          bg: 'bg-emerald-500/10',
          text: 'text-emerald-400',
          border: 'border-emerald-500/20',
          icon: CheckCircle2,
        };
      case 'ACCEPTABLE':
        return {
          stroke: '#3b82f6',
          bg: 'bg-blue-500/10',
          text: 'text-blue-400',
          border: 'border-blue-500/20',
          icon: ShieldCheck,
        };
      case 'POOR':
        return {
          stroke: '#f59e0b',
          bg: 'bg-amber-500/10',
          text: 'text-amber-400',
          border: 'border-amber-500/20',
          icon: AlertTriangle,
        };
      case 'UNUSABLE':
      default:
        return {
          stroke: '#ef4444',
          bg: 'bg-rose-500/10',
          text: 'text-rose-400',
          border: 'border-rose-500/20',
          icon: XCircle,
        };
    }
  };

  const config = getColorConfig();
  const Icon = config.icon;

  return (
    <div className="glass-panel rounded-2xl p-6 flex flex-col items-center justify-center relative overflow-hidden">
      <div className="relative w-40 h-40 flex items-center justify-center">
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="80"
            cy="80"
            r={radius}
            className="text-slate-800"
            strokeWidth="10"
            stroke="currentColor"
            fill="transparent"
          />
          <circle
            cx="80"
            cy="80"
            r={radius}
            stroke={config.stroke}
            strokeWidth="10"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            fill="transparent"
            className="transition-all duration-1000 ease-out"
          />
        </svg>

        <div className="absolute flex flex-col items-center justify-center text-center">
          <span className="text-4xl font-extrabold tracking-tight text-white font-mono">
            {score.toFixed(1)}
          </span>
          <span className="text-[11px] font-semibold text-muted-foreground uppercase tracking-widest">
            / 100
          </span>
        </div>
      </div>

      <div className="mt-4 flex flex-col items-center">
        <div
          className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${config.bg} ${config.text} border ${config.border}`}
        >
          <Icon className="w-3.5 h-3.5" />
          <span>{label}</span>
        </div>
        {processingTimeMs !== undefined && (
          <span className="text-[11px] font-mono text-muted-foreground mt-2">
            Inference: {processingTimeMs.toFixed(1)}ms
          </span>
        )}
      </div>
    </div>
  );
};

