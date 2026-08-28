import React from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts";
import { Activity } from "lucide-react";

interface FeatureRadarProps {
  stats: Record<string, number>;
  qualityScore: number;
}

export const FeatureRadar: React.FC<FeatureRadarProps> = ({ stats, qualityScore }) => {
  // Normalize key measurements to 0-100 index for radar visualization
  const getNormalizedMetrics = () => {
    const laplacian = stats["sharpness_laplacian"] || 0;
    const sharpnessScore = Math.min(100, Math.max(0, (laplacian / 300) * 100));

    const meanLuma = stats["exposure_mean_luma"] || 128;
    const exposureDist = Math.abs(meanLuma - 128);
    const exposureScore = Math.max(0, 100 - (exposureDist / 128) * 100);

    const noiseFloor = stats["noise_flat_sigma"] || 0;
    const noiseScore = Math.max(0, 100 - (noiseFloor / 15) * 100);

    const blockiness = stats["artifacts_blockiness"] || 1.0;
    const blockDist = Math.abs(blockiness - 1.0);
    const compressionScore = Math.max(0, 100 - (blockDist / 0.5) * 100);

    const contrast = stats["exposure_rms_contrast"] || 0;
    const contrastScore = Math.min(100, Math.max(0, (contrast / 60) * 100));

    const linear = stats["defects_linear_structure"] || 0;
    const structuralScore = Math.max(0, 100 - (linear / 0.3) * 100);

    return [
      { subject: "Sharpness", value: Math.round(sharpnessScore) },
      { subject: "Exposure", value: Math.round(exposureScore) },
      { subject: "Clean Floor", value: Math.round(noiseScore) },
      { subject: "Contrast", value: Math.round(contrastScore) },
      { subject: "Integrity", value: Math.round(structuralScore) },
      { subject: "Compression", value: Math.round(compressionScore) },
    ];
  };

  const data = getNormalizedMetrics();

  return (
    <div className="glass-card rounded-2xl p-5 flex flex-col items-center justify-between h-full">
      <div className="w-full flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-indigo-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Quality Balance Radar
          </h3>
        </div>
        <span className="text-[10px] font-mono text-slate-400">
          6-Axis Normalized
        </span>
      </div>

      <div className="w-full h-56 flex items-center justify-center">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
            <PolarGrid stroke="#334155" />
            <PolarAngleAxis
              dataKey="subject"
              tick={{ fill: "#94a3b8", fontSize: 10, fontFamily: "Plus Jakarta Sans" }}
            />
            <PolarRadiusAxis
              angle={30}
              domain={[0, 100]}
              tick={{ fill: "#64748b", fontSize: 9 }}
              stroke="#1e293b"
            />
            <Radar
              name="Quality Index"
              dataKey="value"
              stroke="#6366f1"
              fill="#6366f1"
              fillOpacity={0.4}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-[11px] text-slate-400 text-center mt-1">
        Balanced geometric & radiometric profile across 6 key physical axes.
      </p>
    </div>
  );
};

