import React from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from 'recharts';

interface FeatureRadarProps {
  stats: Record<string, number>;
}

export const FeatureRadar: React.FC<FeatureRadarProps> = ({ stats }) => {
  // Normalize key indicators to 0-100 scale for visual radar representation
  const sharpness = Math.min(100, Math.max(0, (stats['sharpness_laplacian'] || 0) / 3));
  const exposure = Math.min(100, Math.max(0, 100 - Math.abs((stats['exposure_mean_luma'] || 128) - 128) * 0.8));
  const noiseFloor = Math.min(100, Math.max(0, 100 - (stats['noise_sigma_flat'] || 0) * 8));
  const contrast = Math.min(100, Math.max(0, (stats['exposure_contrast_rms'] || 40) * 1.5));
  const integrity = Math.min(100, Math.max(0, 100 - (stats['artifacts_blockiness'] ? (stats['artifacts_blockiness'] - 1.0) * 120 : 0)));
  const geometry = Math.min(100, Math.max(0, (stats['defect_radial_falloff'] || 1.0) * 100));

  const data = [
    { subject: 'Sharpness', value: Math.round(sharpness), fullMark: 100 },
    { subject: 'Exposure', value: Math.round(exposure), fullMark: 100 },
    { subject: 'Noise Floor', value: Math.round(noiseFloor), fullMark: 100 },
    { subject: 'Contrast', value: Math.round(contrast), fullMark: 100 },
    { subject: 'Integrity', value: Math.round(integrity), fullMark: 100 },
    { subject: 'Uniformity', value: Math.round(geometry), fullMark: 100 },
  ];

  return (
    <div className="glass-panel rounded-2xl p-5 flex flex-col items-center">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground self-start mb-2">
        6-Axis Quality Balance
      </h3>
      <div className="w-full h-56">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="75%" data={data}>
            <PolarGrid stroke="#334155" />
            <PolarAngleAxis dataKey="subject" stroke="#94a3b8" tick={{ fontSize: 11 }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#475569" tick={false} />
            <Radar
              name="Quality Score"
              dataKey="value"
              stroke="#6366f1"
              fill="#6366f1"
              fillOpacity={0.4}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

