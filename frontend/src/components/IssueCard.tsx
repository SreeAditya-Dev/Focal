import React from 'react';
import {
  AlertCircle,
  EyeOff,
  Sun,
  Moon,
  Volume2,
  FileWarning,
  Flame,
  ChevronRight,
} from 'lucide-react';
import { DetectedIssue } from '../api/types';

interface IssueCardProps {
  issue: DetectedIssue;
}

export const IssueCard: React.FC<IssueCardProps> = ({ issue }) => {
  const getIcon = () => {
    switch (issue.type) {
      case 'blur':
        return EyeOff;
      case 'overexposure':
        return Sun;
      case 'underexposure':
        return Moon;
      case 'noise':
        return Volume2;
      case 'corruption':
        return FileWarning;
      case 'defect':
      default:
        return Flame;
    }
  };

  const getSeverityBadge = () => {
    switch (issue.severity) {
      case 'high':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/30';
      case 'medium':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/30';
      case 'low':
      default:
        return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
    }
  };

  const Icon = getIcon();

  return (
    <div className="glass-card rounded-xl p-4 border border-slate-800 hover:border-slate-700 transition-all">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-slate-800/80 flex items-center justify-center text-indigo-400 border border-slate-700/50">
            <Icon className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-sm font-bold text-slate-100 capitalize">{issue.type}</h4>
            <div className="flex items-center gap-2 mt-0.5">
              <span
                className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded border ${getSeverityBadge()}`}
              >
                {issue.severity} severity
              </span>
              <span className="text-xs font-mono text-slate-400">
                {(issue.confidence * 100).toFixed(0)}% confidence
              </span>
            </div>
          </div>
        </div>

        <div className="text-right font-mono text-xs text-slate-400">
          <div>Rule: {(issue.rule_confidence * 100).toFixed(0)}%</div>
          <div>CNN: {(issue.cnn_confidence * 100).toFixed(0)}%</div>
        </div>
      </div>

      {issue.evidence && issue.evidence.length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-800/60 space-y-1">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
            Physical Evidence:
          </span>
          {issue.evidence.map((ev, i) => (
            <div key={i} className="flex items-start gap-1.5 text-xs text-slate-300">
              <ChevronRight className="w-3.5 h-3.5 text-indigo-400 shrink-0 mt-0.5" />
              <span>{ev}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

