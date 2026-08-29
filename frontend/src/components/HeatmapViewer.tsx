import React, { useState } from 'react';
import { Layers, Sliders, Eye, EyeOff } from 'lucide-react';

interface HeatmapViewerProps {
  originalImage: string | null;
  heatmapBase64?: string | null;
  heatmapIssue?: string | null;
}

export const HeatmapViewer: React.FC<HeatmapViewerProps> = ({
  originalImage,
  heatmapBase64,
  heatmapIssue,
}) => {
  const [opacity, setOpacity] = useState(0.65);
  const [mode, setMode] = useState<'overlay' | 'side-by-side'>('overlay');

  if (!originalImage) return null;

  return (
    <div className="glass-panel rounded-2xl p-5 overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-bold text-foreground">Grad-CAM Explainability Map</h3>
          {heatmapIssue && (
            <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              Target: {heatmapIssue}
            </span>
          )}
        </div>

        {heatmapBase64 && (
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2 bg-card px-2 py-1 rounded-lg border border-border">
              <button
                onClick={() => setMode('overlay')}
                className={`px-2 py-0.5 rounded font-medium transition-colors ${
                  mode === 'overlay' ? 'bg-indigo-600 text-white' : 'text-muted-foreground hover:text-white'
                }`}
              >
                Overlay
              </button>
              <button
                onClick={() => setMode('side-by-side')}
                className={`px-2 py-0.5 rounded font-medium transition-colors ${
                  mode === 'side-by-side' ? 'bg-indigo-600 text-white' : 'text-muted-foreground hover:text-white'
                }`}
              >
                Side-by-Side
              </button>
            </div>

            {mode === 'overlay' && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Sliders className="w-3.5 h-3.5" />
                <span>Heatmap: {(opacity * 100).toFixed(0)}%</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-20 accent-indigo-500 cursor-pointer"
                />
              </div>
            )}
          </div>
        )}
      </div>

      <div className="relative rounded-xl overflow-hidden bg-card/80 border border-border flex items-center justify-center min-h-[300px]">
        {mode === 'side-by-side' && heatmapBase64 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 w-full p-2">
            <div className="flex flex-col items-center">
              <span className="text-[11px] font-semibold text-muted-foreground mb-1">Original Image</span>
              <img
                src={originalImage}
                alt="Original"
                className="max-h-[380px] w-auto object-contain rounded-lg shadow-md"
              />
            </div>
            <div className="flex flex-col items-center">
              <span className="text-[11px] font-semibold text-muted-foreground mb-1">Grad-CAM Activation</span>
              <img
                src={`data:image/png;base64,${heatmapBase64}`}
                alt="Heatmap"
                className="max-h-[380px] w-auto object-contain rounded-lg shadow-md"
              />
            </div>
          </div>
        ) : (
          <div className="relative max-h-[420px] w-full flex items-center justify-center p-2">
            <img
              src={originalImage}
              alt="Source preview"
              className="max-h-[400px] w-auto object-contain rounded-lg"
            />
            {heatmapBase64 && (
              <img
                src={`data:image/png;base64,${heatmapBase64}`}
                alt="Grad-CAM Overlay"
                style={{ opacity }}
                className="absolute max-h-[400px] w-auto object-contain rounded-lg transition-opacity duration-150 pointer-events-none"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

