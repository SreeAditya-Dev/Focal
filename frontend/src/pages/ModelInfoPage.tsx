import React from 'react';
import { Cpu, ShieldCheck, Activity, Layers, BarChart2, Zap } from 'lucide-react';

export const ModelInfoPage: React.FC = () => {
  const metrics = [
    { name: 'Macro ROC-AUC', value: '94.5%', benchmark: 'Held-Out Test Set (3,130 images)' },
    { name: 'Macro Recall', value: '91.2%', benchmark: 'Across all 6 defect classes' },
    { name: 'Exact Band Accuracy', value: '53.3%', benchmark: '4-tier classification' },
    { name: 'Within-1-Band Accuracy', value: '94.8%', benchmark: 'Near-neighbor classification' },
    { name: 'Quality Score MAE', value: '11.62', benchmark: 'Points error vs 21.95 baseline' },
    { name: 'CPU Inference Latency', value: '130ms', benchmark: 'Full-resolution image forward pass' },
  ];

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div>
        <h2 className="text-xl font-bold text-slate-100">Focal Architecture & Benchmarks</h2>
        <p className="text-xs text-slate-400 mt-0.5">
          Dual-branch hybrid computer vision and deep learning quality assessment pipeline
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {metrics.map((m, idx) => (
          <div key={idx} className="glass-panel rounded-xl p-4 border border-slate-800">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
              {m.name}
            </span>
            <span className="text-2xl font-extrabold font-mono text-indigo-400 mt-1 block">
              {m.value}
            </span>
            <span className="text-xs text-slate-500 mt-1 block">{m.benchmark}</span>
          </div>
        ))}
      </div>

      <div className="glass-panel rounded-2xl p-6 space-y-4">
        <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-400" />
          The Dual-Branch Hybrid Approach
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          Standard deep convolutional neural networks downsample images to 224×224 px, which completely obliterates 1-pixel sensor noise, thin hairline scratches, and subtle JPEG DCT block edges. Focal solves this with an asymmetric dual branch:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          <div className="glass-card rounded-xl p-4 border border-slate-800">
            <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wider mb-1 flex items-center gap-1.5">
              <Cpu className="w-4 h-4" /> Branch 1: Full-Res Classical CV
            </h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Extracts 47 spatial and frequency domain features (Laplacian variance, Tenengrad energy, Immerkaer noise mask, flattest-block noise floor, GLCM texture contrast, and 8×8 DCT grid blockiness) on the full-resolution image.
            </p>
          </div>

          <div className="glass-card rounded-xl p-4 border border-slate-800">
            <h4 className="text-xs font-bold text-purple-300 uppercase tracking-wider mb-1 flex items-center gap-1.5">
              <Zap className="w-4 h-4" /> Branch 2: MobileNetV3 CNN + MLP
            </h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Feeds both the 224×224 RGB image and the 47-feature MLP projection vector into a joint multi-task neural network with continuous presence and severity heads.
            </p>
          </div>
        </div>
      </div>

      <div className="glass-panel rounded-2xl p-6 space-y-3">
        <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-indigo-400" />
          Confidence Calibration & Explainable AI
        </h3>
        <ul className="list-disc list-inside text-xs text-slate-300 space-y-2 leading-relaxed">
          <li>
            <strong className="text-slate-100">Vector Temperature Scaling:</strong> Calibrates neural network logits to empirical frequencies, bounding Expected Calibration Error (ECE) below 4.2%.
          </li>
          <li>
            <strong className="text-slate-100">Grad-CAM Heatmap Overlays:</strong> Hooks into the final convolutional layer activations to localize exactly where visual defects or blur regions occur.
          </li>
          <li>
            <strong className="text-slate-100">Monte-Carlo Dropout:</strong> Executes 20 stochastic forward passes to surface uncertainty on ambiguous edge-case images.
          </li>
        </ul>
      </div>
    </div>
  );
};
