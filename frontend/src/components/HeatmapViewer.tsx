import React, { useState } from "react";
import { Eye, Layers, Maximize2, Sparkles } from "lucide-react";

interface HeatmapViewerProps {
  originalImage: string;
  heatmapBase64?: string | null;
  heatmapIssue?: string | null;
}

export const HeatmapViewer: React.FC<HeatmapViewerProps> = ({
  originalImage,
  heatmapBase64,
  heatmapIssue,
}) => {
  const [viewMode, setViewMode] = useState<"overlay" | "split" | "original">("overlay");
  const [opacity, setOpacity] = useState<number>(0.65);

  const heatmapSrc = heatmapBase64
    ? heatmapBase64.startsWith("data:")
      ? heatmapBase64
      : `data:image/jpeg;base64,${heatmapBase64}`
    : null;

  return (
    <div className="glass-card rounded-2xl p-5 overflow-hidden flex flex-col h-full">
      {/* Top Bar */}
      <div className="flex items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-indigo-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Explainability Visualizer
          </h3>
          {heatmapIssue && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-300">
              Grad-CAM: {heatmapIssue}
            </span>
          )}
        </div>

        {/* View Mode Switcher */}
        {heatmapSrc && (
          <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800 text-[11px]">
            <button
              onClick={() => setViewMode("overlay")}
              className={`px-2.5 py-0.5 rounded font-medium transition-all ${
                viewMode === "overlay"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Overlay
            </button>
            <button
              onClick={() => setViewMode("split")}
              className={`px-2.5 py-0.5 rounded font-medium transition-all ${
                viewMode === "split"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setViewMode("original")}
              className={`px-2.5 py-0.5 rounded font-medium transition-all ${
                viewMode === "original"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Original
            </button>
          </div>
        )}
      </div>

      {/* Image Display Area */}
      <div className="relative flex-1 min-h-[300px] w-full rounded-xl overflow-hidden bg-slate-950/80 border border-slate-800/80 flex items-center justify-center">
        {viewMode === "split" && heatmapSrc ? (
          <div className="grid grid-cols-2 gap-2 h-full w-full p-2">
            <div className="relative flex flex-col items-center justify-center rounded-lg overflow-hidden border border-slate-800">
              <img
                src={originalImage}
                alt="Original"
                className="max-h-[360px] w-full object-contain"
              />
              <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-slate-950/80 text-[10px] font-mono text-slate-300 border border-slate-800">
                Original Image
              </span>
            </div>
            <div className="relative flex flex-col items-center justify-center rounded-lg overflow-hidden border border-slate-800">
              <img
                src={heatmapSrc}
                alt="Grad-CAM Heatmap"
                className="max-h-[360px] w-full object-contain"
              />
              <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-slate-950/80 text-[10px] font-mono text-indigo-300 border border-indigo-500/30">
                Grad-CAM Activation
              </span>
            </div>
          </div>
        ) : (
          <div className="relative flex items-center justify-center h-full w-full p-2">
            <img
              src={originalImage}
              alt="Analyzed Frame"
              className="max-h-[380px] w-full object-contain rounded-lg shadow-xl"
            />

            {/* Heatmap Overlay */}
            {viewMode === "overlay" && heatmapSrc && (
              <img
                src={heatmapSrc}
                alt="Grad-CAM Overlay"
                style={{ opacity }}
                className="absolute inset-0 max-h-[380px] w-full object-contain pointer-events-none rounded-lg transition-opacity"
              />
            )}
          </div>
        )}
      </div>

      {/* Opacity Slider Control */}
      {viewMode === "overlay" && heatmapSrc && (
        <div className="flex items-center gap-3 mt-3 px-2 text-xs">
          <span className="text-[11px] font-mono text-slate-400">Heatmap Intensity:</span>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="flex-1 accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
          <span className="text-[11px] font-mono text-slate-300 w-10 text-right">
            {(opacity * 100).toFixed(0)}%
          </span>
        </div>
      )}
    </div>
  );
};

