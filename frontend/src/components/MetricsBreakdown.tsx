import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Cpu } from 'lucide-react';

interface MetricsBreakdownProps {
  stats: Record<string, number>;
}

export const MetricsBreakdown: React.FC<MetricsBreakdownProps> = ({ stats }) => {
  const [isOpen, setIsOpen] = useState(false);

  const categories: Record<string, Array<{ key: string; name: string; unit?: string }>> = {
    'Sharpness & Focus': [
      { key: 'sharpness_laplacian', name: 'Laplacian Variance' },
      { key: 'sharpness_tenengrad', name: 'Tenengrad Energy' },
      { key: 'sharpness_hf_energy_ratio', name: 'FFT High-Freq Ratio' },
      { key: 'sharpness_canny_density', name: 'Canny Edge Density' },
    ],
    'Exposure & Dynamic Range': [
      { key: 'exposure_mean_luma', name: 'Mean Luminance' },
      { key: 'exposure_shadow_clip', name: 'Shadow Clipping', unit: '%' },
      { key: 'exposure_highlight_clip', name: 'Highlight Clipping', unit: '%' },
      { key: 'exposure_contrast_rms', name: 'RMS Contrast' },
    ],
    'Noise & Texture': [
      { key: 'noise_sigma_flat', name: 'Flattest Block σ' },
      { key: 'noise_sigma_immerkaer', name: 'Immerkaer Noise Mask' },
      { key: 'noise_sigma_chroma', name: 'Chroma Noise' },
      { key: 'texture_glcm_contrast', name: 'GLCM Texture Contrast' },
    ],
    'Defects & Artifacts': [
      { key: 'artifacts_blockiness', name: 'JPEG Blockiness Ratio' },
      { key: 'defect_radial_falloff', name: 'Radial Falloff Ratio' },
      { key: 'defect_linear_structure', name: 'Linear Scratch Score' },
      { key: 'defect_contrast_spread', name: 'Local Contrast Spread' },
    ],
  };

  return (
    <div className="glass-panel rounded-2xl p-5">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-bold text-foreground">47 Classical Computer Vision Metrics</h3>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-indigo-400 font-semibold">
          <span>{isOpen ? 'Collapse' : 'Inspect Features'}</span>
          {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {isOpen && (
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 pt-3 border-t border-border/80">
          {Object.entries(categories).map(([catName, metrics]) => (
            <div key={catName} className="glass-card rounded-xl p-3 border border-border/60">
              <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wider mb-2">
                {catName}
              </h4>
              <div className="space-y-1.5 font-mono text-xs">
                {metrics.map((m) => {
                  const val = stats[m.key];
                  return (
                    <div key={m.key} className="flex items-center justify-between py-0.5 text-popover-foreground">
                      <span className="text-muted-foreground font-sans">{m.name}:</span>
                      <span className="font-semibold text-foreground">
                        {val !== undefined ? val.toFixed(3) : '—'} {m.unit || ''}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

