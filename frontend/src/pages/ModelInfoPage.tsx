import React from "react";
import { Cpu, Award, Zap, ShieldCheck, CheckCircle2, BarChart2, Layers, Sliders } from "lucide-react";

export const ModelInfoPage: React.FC = () => {
  return (
    <div className="space-y-8 pb-16 max-w-5xl mx-auto">
      {/* Title */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold">
          <Cpu className="h-3.5 w-3.5" />
          <span>Focal Deep Learning & Classical CV Architecture</span>
        </div>
        <h2 className="text-2xl font-extrabold text-white">
          Hybrid Dual-Branch Vision System
        </h2>
        <p className="text-xs text-slate-400 leading-relaxed max-w-3xl">
          Focal merges 47 mathematically rigorous computer vision features (Laplacian gradient energy, flattest-block noise floor, Immerkaer sigma, GLCM homogeneity, JPEG blockiness) with a transfer-learned MobileNetV3 convolutional neural network.
        </p>
      </div>

      {/* Architecture Highlights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass-card rounded-xl p-5 border border-slate-800 space-y-2">
          <div className="h-8 w-8 rounded-lg bg-sky-500/10 border border-sky-500/30 flex items-center justify-center text-sky-400">
            <Layers className="h-4 w-4" />
          </div>
          <h3 className="text-sm font-bold text-slate-100">Dual-Branch Backbone</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Preserves fine high-frequency defects through full-resolution spatial features while extracting global semantic representations via MobileNetV3.
          </p>
        </div>

        <div className="glass-card rounded-xl p-5 border border-slate-800 space-y-2">
          <div className="h-8 w-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Sliders className="h-4 w-4" />
          </div>
          <h3 className="text-sm font-bold text-slate-100">Calibrated Fusion</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Temperature-scaled probability logits fused with deterministic heuristic rules to eliminate hallucination on edge cases.
          </p>
        </div>

        <div className="glass-card rounded-xl p-5 border border-slate-800 space-y-2">
          <div className="h-8 w-8 rounded-lg bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
            <Zap className="h-4 w-4" />
          </div>
          <h3 className="text-sm font-bold text-slate-100">Real-Time Latency</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Optimized ~130ms CPU forward inference per frame with integrated Grad-CAM heatmap localization and MC Dropout uncertainty estimation.
          </p>
        </div>
      </div>

      {/* Test Split Benchmark Table */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-white">
              Held-Out Test Set Performance (3,130 Images)
            </h3>
            <p className="text-xs text-slate-400">
              Evaluated against unseen synthetic and natural degradation benchmarks.
            </p>
          </div>
          <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-bold">
            94.5% Macro ROC-AUC
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 text-slate-400 font-mono text-[11px] border-b border-slate-800 uppercase">
              <tr>
                <th className="py-2.5 px-3">Degradation Category</th>
                <th className="py-2.5 px-3">Precision</th>
                <th className="py-2.5 px-3">Recall</th>
                <th className="py-2.5 px-3">F1-Score</th>
                <th className="py-2.5 px-3">ROC-AUC</th>
                <th className="py-2.5 px-3">PR-AUC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50 font-mono">
              <tr className="hover:bg-slate-800/20">
                <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Blur / Insufficient Sharpness</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.968</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.974</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.971</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.998</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.993</td>
              </tr>
              <tr className="hover:bg-slate-800/20">
                <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Overexposure (Highlights)</td>
                <td className="py-2.5 px-3 text-slate-300">0.841</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.926</td>
                <td className="py-2.5 px-3 text-slate-300">0.881</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.990</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.945</td>
              </tr>
              <tr className="hover:bg-slate-800/20">
                <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Image Noise / Sensor Grain</td>
                <td className="py-2.5 px-3 text-slate-300">0.782</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.964</td>
                <td className="py-2.5 px-3 text-slate-300">0.864</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.987</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.935</td>
              </tr>
              <tr className="hover:bg-slate-800/20">
                <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Underexposure (Shadows)</td>
                <td className="py-2.5 px-3 text-slate-300">0.603</td>
                <td className="py-2.5 px-3 text-slate-300">0.862</td>
                <td className="py-2.5 px-3 text-slate-300">0.710</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.958</td>
                <td className="py-2.5 px-3 text-slate-300">0.835</td>
              </tr>
              <tr className="hover:bg-slate-800/20">
                <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Compression Artifacts</td>
                <td className="py-2.5 px-3 text-slate-300">0.595</td>
                <td className="py-2.5 px-3 text-slate-300">0.898</td>
                <td className="py-2.5 px-3 text-slate-300">0.716</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">0.946</td>
                <td className="py-2.5 px-3 text-slate-300">0.865</td>
              </tr>
              <tr className="hover:bg-slate-800/20">
                <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Localized Visual Defects</td>
                <td className="py-2.5 px-3 text-slate-300">0.246</td>
                <td className="py-2.5 px-3 text-slate-300">0.845</td>
                <td className="py-2.5 px-3 text-slate-300">0.381</td>
                <td className="py-2.5 px-3 text-slate-300">0.792</td>
                <td className="py-2.5 px-3 text-slate-300">0.410</td>
              </tr>
              <tr className="bg-indigo-500/10 font-bold">
                <td className="py-3 px-3 font-sans text-indigo-300">MACRO AVERAGE TOTAL</td>
                <td className="py-3 px-3 text-indigo-300">0.672</td>
                <td className="py-3 px-3 text-indigo-300">0.912</td>
                <td className="py-3 px-3 text-indigo-300">0.754</td>
                <td className="py-3 px-3 text-indigo-300">0.945</td>
                <td className="py-3 px-3 text-indigo-300">0.830</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

