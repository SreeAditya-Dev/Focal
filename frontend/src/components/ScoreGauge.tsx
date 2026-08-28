import React from "react";
import { QualityLabel } from "../api/types";
import { ShieldCheck, AlertTriangle, XCircle, Award } from "lucide-react";

interface ScoreGaugeProps {
  score: number;
  label: QualityLabel;
  processingTimeMs?: number;
  uncertainty?: number | null;
}

export const ScoreGauge: React.FC<ScoreGaugeProps> = ({
  score,
  label,
  processingTimeMs,
  uncertainty,
}) => {
  // Score-based coloring
  const getColor = (s: number) => {
    if (s >= 85) return { stroke: "#10b981", text: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/30" };
    if (s >= 70) return { stroke: "#3b82f6", text: "text-blue-400", bg: "bg-blue-500/10", border: "border-blue-500/30" };
    if (s >= 40) return { stroke: "#f59e0b", text: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/30" };
    return { stroke: "#f43f5e", text: "text-rose-400", bg: "bg-rose-500/10", border: "border-rose-500/30" };
  };

  const color = getColor(score);

  // SVG circular arc math
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  const getLabelIcon = () => {
    switch (label) {
      case "EXCELLENT":
        return <Award className="h-4 w-4" />;
      case "ACCEPTABLE":
        return <ShieldCheck className="h-4 w-4" />;
      case "POOR":
        return <AlertTriangle className="h-4 w-4" />;
      case "UNUSABLE":
        return <XCircle className="h-4 w-4" />;
    }
  };

  return (
    <div className="glass-card rounded-2xl p-6 flex flex-col items-center justify-between text-center relative overflow-hidden h-full">
      {/* Background radial highlight */}
      <div
        className="absolute inset-0 opacity-10 blur-2xl pointer-events-none rounded-full"
        style={{ backgroundColor: color.stroke }}
      />

      <div className="w-full flex items-center justify-between text-xs text-slate-400 mb-2">
        <span className="font-bold uppercase tracking-wider text-[11px] text-slate-300">
          Overall Quality Score
        </span>
        {processingTimeMs !== undefined && (
          <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
            {processingTimeMs.toFixed(1)} ms
          </span>
        )}
      </div>

      {/* SVG Radial Gauge */}
      <div className="relative my-2 flex items-center justify-center">
        <svg className="w-44 h-44 -rotate-90 transform" viewBox="0 0 160 160">
          {/* Background circle */}
          <circle
            cx="80"
            cy="80"
            r={radius}
            stroke="currentColor"
            strokeWidth="10"
            className="text-slate-800/80"
            fill="transparent"
          />
          {/* Progress Arc */}
          <circle
            cx="80"
            cy="80"
            r={radius}
            stroke={color.stroke}
            strokeWidth="10"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
            fill="transparent"
          />
        </svg>

        {/* Center Score readout */}
        <div className="absolute flex flex-col items-center justify-center">
          <span className={`text-4xl font-extrabold tracking-tight ${color.text}`}>
            {score.toFixed(1)}
          </span>
          <span className="text-[11px] uppercase font-mono text-slate-400 mt-0.5">
            out of 100
          </span>
        </div>
      </div>

      {/* Quality Badge */}
      <div className="w-full mt-2 space-y-2">
        <div
          className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border ${color.bg} ${color.border} ${color.text}`}
        >
          {getLabelIcon()}
          <span>{label}</span>
        </div>

        {uncertainty !== undefined && uncertainty !== null && (
          <div className="text-[11px] font-mono text-slate-400 flex items-center justify-center gap-1">
            <span>Model Uncertainty:</span>
            <span className="text-slate-200 font-semibold">
              ±{(uncertainty * 100).toFixed(1)}%
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

