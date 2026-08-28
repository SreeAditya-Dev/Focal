import React, { useState } from "react";
import { ChevronDown, ChevronUp, BarChart2, Zap } from "lucide-react";

interface MetricsBreakdownProps {
  stats: Record<string, number>;
}

export const MetricsBreakdown: React.FC<MetricsBreakdownProps> = ({ stats }) => {
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    sharpness: true,
    exposure: true,
    noise: true,
    artifacts: false,
    defects: false,
  });

  const toggleSection = (section: string) => {
    setOpenSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const groups = [
    {
      id: "sharpness",
      title: "Sharpness & High-Frequency Detail",
      keys: [
        { key: "sharpness_laplacian", label: "Laplacian Variance", threshold: "≥ 150 (Clean)" },
        { key: "sharpness_tenengrad", label: "Tenengrad Gradient Energy", threshold: "Higher = Sharper" },
        { key: "sharpness_fft_high_ratio", label: "FFT High-Frequency Ratio", threshold: "0.02 - 0.15" },
        { key: "sharpness_canny_density", label: "Canny Edge Density", threshold: "0.05 - 0.25" },
      ],
    },
    {
      id: "exposure",
      title: "Exposure & Dynamic Range",
      keys: [
        { key: "exposure_mean_luma", label: "Mean Luminance (V-channel)", threshold: "70 - 185" },
        { key: "exposure_shadow_clipping", label: "Shadow Pixel Clipping", threshold: "< 2.0%" },
        { key: "exposure_highlight_clipping", label: "Highlight Pixel Clipping", threshold: "< 1.5%" },
        { key: "exposure_rms_contrast", label: "RMS Global Contrast", threshold: "≥ 25" },
      ],
    },
    {
      id: "noise",
      title: "Noise Floor & Sensor Grain",
      keys: [
        { key: "noise_flat_sigma", label: "Flattest Block Noise Floor (σ)", threshold: "< 4.5" },
        { key: "noise_immerkaer", label: "Immerkaer Fast Noise Estimator", threshold: "< 6.0" },
        { key: "noise_chroma_sigma", label: "Chroma Variation Noise (σ)", threshold: "< 3.5" },
        { key: "noise_impulse_ratio", label: "Impulse Salt-and-Pepper Ratio", threshold: "< 0.005" },
      ],
    },
    {
      id: "artifacts",
      title: "Compression & Blockiness Artifacts",
      keys: [
        { key: "artifacts_blockiness", label: "JPEG 8×8 Block Grid Ratio", threshold: "~1.00 (No blocks)" },
        { key: "artifacts_flat_block_frac", label: "Flat Block Fraction", threshold: "< 0.40" },
        { key: "artifacts_byte_entropy", label: "Raw Byte Stream Entropy", threshold: "7.2 - 7.99" },
      ],
    },
    {
      id: "defects",
      title: "Geometric Defects & Smudges",
      keys: [
        { key: "defects_radial_falloff", label: "Radial Corner Falloff (Vignette)", threshold: "< 0.25" },
        { key: "defects_linear_structure", label: "Linear Scratch Structure Length", threshold: "< 0.15" },
        { key: "defects_local_contrast_spread", label: "Per-Tile Local Contrast Spread", threshold: "< 0.35" },
      ],
    },
  ];

  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="flex items-center gap-2 mb-4">
        <BarChart2 className="h-4 w-4 text-indigo-400" />
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
          Classical CV Measurements (47 Extracted Features)
        </h3>
      </div>

      <div className="space-y-3">
        {groups.map((group) => {
          const isOpen = openSections[group.id];
          return (
            <div
              key={group.id}
              className="rounded-xl border border-slate-800 bg-slate-900/40 overflow-hidden"
            >
              <button
                onClick={() => toggleSection(group.id)}
                className="w-full px-4 py-2.5 flex items-center justify-between text-left hover:bg-slate-800/40 transition-colors"
              >
                <span className="text-xs font-semibold text-slate-200">
                  {group.title}
                </span>
                {isOpen ? (
                  <ChevronUp className="h-3.5 w-3.5 text-slate-400" />
                ) : (
                  <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
                )}
              </button>

              {isOpen && (
                <div className="px-4 pb-3 pt-1 divide-y divide-slate-800/60">
                  {group.keys.map((item) => {
                    const val = stats[item.key];
                    return (
                      <div
                        key={item.key}
                        className="py-2 flex items-center justify-between text-xs font-mono"
                      >
                        <div>
                          <span className="text-slate-300 font-sans block text-xs">
                            {item.label}
                          </span>
                          <span className="text-[10px] text-slate-500 font-sans">
                            Reference: {item.threshold}
                          </span>
                        </div>
                        <span className="font-bold text-indigo-300">
                          {val !== undefined ? val.toFixed(3) : "—"}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

