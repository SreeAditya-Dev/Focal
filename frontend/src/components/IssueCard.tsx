import React from "react";
import { Issue } from "../api/types";
import {
  EyeOff,
  Sun,
  Moon,
  Radio,
  Grid,
  Sparkles,
  Info,
  Sliders,
  Cpu,
} from "lucide-react";

interface IssueCardProps {
  issue: Issue;
}

export const IssueCard: React.FC<IssueCardProps> = ({ issue }) => {
  const getIssueMeta = (type: string) => {
    switch (type.toLowerCase()) {
      case "blur":
        return {
          title: "Blur / Insufficient Sharpness",
          icon: <EyeOff className="h-4 w-4 text-sky-400" />,
          color: "border-sky-500/30 bg-sky-500/10 text-sky-400",
        };
      case "overexposure":
        return {
          title: "Overexposure (Blown Highlights)",
          icon: <Sun className="h-4 w-4 text-amber-400" />,
          color: "border-amber-500/30 bg-amber-500/10 text-amber-400",
        };
      case "underexposure":
        return {
          title: "Underexposure (Shadow Crush)",
          icon: <Moon className="h-4 w-4 text-indigo-400" />,
          color: "border-indigo-500/30 bg-indigo-500/10 text-indigo-400",
        };
      case "noise":
        return {
          title: "High Image Noise / Grain",
          icon: <Radio className="h-4 w-4 text-purple-400" />,
          color: "border-purple-500/30 bg-purple-500/10 text-purple-400",
        };
      case "corruption":
        return {
          title: "Compression Artifacts / Glitches",
          icon: <Grid className="h-4 w-4 text-rose-400" />,
          color: "border-rose-500/30 bg-rose-500/10 text-rose-400",
        };
      case "defect":
        return {
          title: "Localized Visual Defect / Smudge",
          icon: <Sparkles className="h-4 w-4 text-emerald-400" />,
          color: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
        };
      default:
        return {
          title: issue.type,
          icon: <Info className="h-4 w-4 text-slate-400" />,
          color: "border-slate-500/30 bg-slate-500/10 text-slate-400",
        };
    }
  };

  const meta = getIssueMeta(issue.type);

  const getSeverityBadge = (sev: string) => {
    switch (sev.toLowerCase()) {
      case "low":
        return "bg-blue-500/10 text-blue-400 border-blue-500/30";
      case "medium":
        return "bg-amber-500/10 text-amber-400 border-amber-500/30";
      case "high":
        return "bg-orange-500/10 text-orange-400 border-orange-500/30";
      case "severe":
        return "bg-rose-500/10 text-rose-400 border-rose-500/30";
      default:
        return "bg-slate-800 text-slate-300 border-slate-700";
    }
  };

  return (
    <div className="glass-card rounded-xl p-4 border border-slate-800/80 hover:border-slate-700 transition-all flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between gap-2 mb-2.5">
          <div className="flex items-center gap-2">
            <div className={`p-1.5 rounded-lg border ${meta.color}`}>
              {meta.icon}
            </div>
            <h4 className="text-xs font-bold text-slate-100">{meta.title}</h4>
          </div>

          <span
            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase border ${getSeverityBadge(
              issue.severity
            )}`}
          >
            {issue.severity}
          </span>
        </div>

        {/* Confidence & Severity Progress */}
        <div className="space-y-1.5 mb-3 bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/50">
          <div className="flex items-center justify-between text-[11px] font-mono">
            <span className="text-slate-400">Detection Confidence</span>
            <span className="font-bold text-slate-200">
              {(issue.confidence * 100).toFixed(0)}%
            </span>
          </div>
          <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, Math.max(0, issue.confidence * 100))}%` }}
            />
          </div>

          {/* Rules vs CNN blend */}
          <div className="flex items-center justify-between text-[10px] text-slate-500 pt-1">
            <span className="flex items-center gap-1">
              <Sliders className="h-3 w-3" /> Rule: {(issue.rule_confidence * 100).toFixed(0)}%
            </span>
            <span className="flex items-center gap-1">
              <Cpu className="h-3 w-3" /> CNN: {(issue.cnn_confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Evidence reasons */}
        {issue.evidence && issue.evidence.length > 0 && (
          <div className="space-y-1">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Heuristic & Feature Evidence:
            </span>
            <ul className="space-y-1">
              {issue.evidence.map((reason, idx) => (
                <li
                  key={idx}
                  className="text-[11px] text-slate-300 flex items-start gap-1.5 leading-tight"
                >
                  <span className="text-indigo-400 font-bold">•</span>
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

