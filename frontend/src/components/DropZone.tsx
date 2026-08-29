import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Sparkles, Layers, FileX } from 'lucide-react';

interface DropZoneProps {
  onFileSelected: (file: File) => void;
  onBatchSelected: (files: File[]) => void;
  isLoading: boolean;
  batchMode: boolean;
  setBatchMode: (val: boolean) => void;
}

export const DropZone: React.FC<DropZoneProps> = ({
  onFileSelected,
  onBatchSelected,
  isLoading,
  batchMode,
  setBatchMode,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith('image/'));
    if (!files.length) return;

    if (batchMode || files.length > 1) {
      if (!batchMode) setBatchMode(true);
      onBatchSelected(files);
    } else {
      onFileSelected(files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []).filter((f) => f.type.startsWith('image/'));
    if (!files.length) return;

    if (batchMode || files.length > 1) {
      onBatchSelected(files);
    } else {
      onFileSelected(files[0]);
    }
  };

  const loadSample = async (presetType: string) => {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (presetType === 'clean') {
      const grad = ctx.createLinearGradient(0, 0, 640, 480);
      grad.addColorStop(0, '#1e293b');
      grad.addColorStop(1, '#0f172a');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 640, 480);
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(320, 240, 100, 0, Math.PI * 2);
      ctx.fill();
    } else if (presetType === 'blur') {
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, 640, 480);
      ctx.filter = 'blur(16px)';
      ctx.fillStyle = '#f43f5e';
      ctx.fillRect(160, 120, 320, 240);
    } else if (presetType === 'noise') {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, 640, 480);
      const imgData = ctx.getImageData(0, 0, 640, 480);
      for (let i = 0; i < imgData.data.length; i += 4) {
        const noise = (Math.random() - 0.5) * 140;
        imgData.data[i] = Math.min(255, Math.max(0, 100 + noise));
        imgData.data[i + 1] = Math.min(255, Math.max(0, 120 + noise));
        imgData.data[i + 2] = Math.min(255, Math.max(0, 150 + noise));
      }
      ctx.putImageData(imgData, 0, 0);
    }

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `sample_${presetType}.jpg`, { type: 'image/jpeg' });
        onFileSelected(file);
      }
    }, 'image/jpeg');
  };

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Upload Image{batchMode ? 's' : ''}
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setBatchMode(!batchMode)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold transition-all ${
              batchMode
                ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                : 'text-muted-foreground hover:text-card-foreground bg-card border border-border'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            {batchMode ? 'Batch Mode Active' : 'Enable Batch Mode'}
          </button>
        </div>
      </div>

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 glass-panel ${
          isDragging
            ? 'border-indigo-500 bg-indigo-500/10 scale-[1.01]'
            : 'border-border hover:border-border hover:bg-secondary/60'
        } ${isLoading ? 'pointer-events-none opacity-50' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple={batchMode}
          onChange={handleFileInput}
          className="hidden"
        />

        <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 mb-4 group-hover:scale-110 transition-transform">
          <UploadCloud className="w-7 h-7" />
        </div>

        <h3 className="text-base font-bold text-foreground mb-1">
          {batchMode ? 'Drop multiple images here' : 'Drop your image here or browse'}
        </h3>
        <p className="text-xs text-muted-foreground max-w-sm mb-4">
          Supports high-resolution JPEG, PNG, WebP, BMP, and TIFF files.
        </p>

        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <ImageIcon className="w-3.5 h-3.5" />
          <span>Full 47-feature CV analysis + Grad-CAM Heatmap</span>
        </div>
      </div>

      {!batchMode && (
        <div className="mt-3 flex items-center gap-2">
          <span className="text-xs text-muted-foreground font-medium">Quick test samples:</span>
          <button
            onClick={() => loadSample('clean')}
            className="px-2 py-1 rounded bg-card hover:bg-muted border border-border text-xs text-popover-foreground font-medium transition-colors"
          >
            Clean Sample
          </button>
          <button
            onClick={() => loadSample('blur')}
            className="px-2 py-1 rounded bg-card hover:bg-muted border border-border text-xs text-rose-300 font-medium transition-colors"
          >
            Blur Sample
          </button>
          <button
            onClick={() => loadSample('noise')}
            className="px-2 py-1 rounded bg-card hover:bg-muted border border-border text-xs text-amber-300 font-medium transition-colors"
          >
            Noisy Sample
          </button>
        </div>
      )}
    </div>
  );
};

