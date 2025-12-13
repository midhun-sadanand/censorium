'use client';

import { useState, useEffect } from 'react';
import { redactImage, downloadBlob, getRedactionMetadata, type RedactionOptions, type Detection } from '@/lib/api';

interface RedactionViewerProps {
  file: File;
  onRemove: () => void;
}

export default function RedactionViewer({ file, onRemove }: RedactionViewerProps) {
  const [originalUrl, setOriginalUrl] = useState<string>('');
  const [redactedUrl, setRedactedUrl] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string>('');
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [detectionCount, setDetectionCount] = useState<number | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  
  // Settings
  const [mode, setMode] = useState<'blur' | 'pixelate'>('blur');
  const [confidence, setConfidence] = useState(0.5);
  const [showSettings, setShowSettings] = useState(false);

  // Load original image
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setOriginalUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Auto-process on mount and when settings change
  useEffect(() => {
    processImage();
  }, [mode, confidence]);

  const processImage = async () => {
    setIsProcessing(true);
    setError('');
    
    try {
      const options: RedactionOptions = {
        mode,
        confidence_threshold: confidence,
        padding_factor: 0.1,
        blur_kernel_size: 51,
        pixelate_block_size: 15
      };

      // Get metadata first
      const metadata = await getRedactionMetadata(file, options);
      setDetectionCount(metadata.detections.length);
      setProcessingTime(metadata.processing_time_ms);
      setDetections(metadata.detections);

      // Get redacted image
      const blob = await redactImage(file, options);
      
      // Clean up previous URL
      if (redactedUrl) {
        URL.revokeObjectURL(redactedUrl);
      }
      
      const url = URL.createObjectURL(blob);
      setRedactedUrl(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process image');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (redactedUrl) {
      const extension = file.name.split('.').pop() || 'jpg';
      const filename = `redacted_${file.name.replace(/\.[^/.]+$/, '')}.${extension}`;
      
      fetch(redactedUrl)
        .then(res => res.blob())
        .then(blob => downloadBlob(blob, filename));
    }
  };

  const faceCount = detections.filter(d => d.entity_type === 'face').length;
  const plateCount = detections.filter(d => d.entity_type === 'license_plate').length;

  return (
    <div 
      className="rounded-lg overflow-hidden"
      style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
    >
      {/* Header */}
      <div 
        className="px-6 py-4 flex items-center justify-between"
        style={{ borderBottom: '1px solid var(--color-border)' }}
      >
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold truncate" style={{ color: 'var(--color-text-primary)' }}>
            {file.name}
          </h3>
          <p className="text-sm mt-1" style={{ color: 'var(--color-text-secondary)' }}>
            {(file.size / 1024 / 1024).toFixed(2)} MB
            {detectionCount !== null && (
              <span className="ml-3">
                {detectionCount} {detectionCount === 1 ? 'detection' : 'detections'}
                {faceCount > 0 && ` • ${faceCount} face${faceCount !== 1 ? 's' : ''}`}
                {plateCount > 0 && ` • ${plateCount} plate${plateCount !== 1 ? 's' : ''}`}
              </span>
            )}
            {processingTime !== null && (
              <span className="ml-3" style={{ color: 'var(--color-accent)' }}>
                {processingTime.toFixed(0)}ms
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2 ml-4">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 rounded-lg transition-colors"
            style={{ color: 'var(--color-text-secondary)' }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'var(--color-bg-tertiary)';
              e.currentTarget.style.color = 'var(--color-text-primary)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
              e.currentTarget.style.color = 'var(--color-text-secondary)';
            }}
            title="Settings"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
          <button
            onClick={handleDownload}
            disabled={!redactedUrl || isProcessing}
            className="px-4 py-2 text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ 
              background: !redactedUrl || isProcessing ? 'var(--color-bg-tertiary)' : 'var(--color-accent)', 
              color: 'var(--color-text-primary)'
            }}
          >
            Download
          </button>
          <button
            onClick={onRemove}
            className="p-2 rounded-lg transition-colors"
            style={{ color: 'var(--color-text-secondary)' }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)';
              e.currentTarget.style.color = 'var(--color-error)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
              e.currentTarget.style.color = 'var(--color-text-secondary)';
            }}
            title="Remove"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div 
          className="px-6 py-4"
          style={{ 
            background: 'var(--color-bg-tertiary)', 
            borderBottom: '1px solid var(--color-border)' 
          }}
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Mode Selection */}
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-primary)' }}>
                Redaction Mode
              </label>
              <div className="flex gap-2">
                <button
                  onClick={() => setMode('blur')}
                  className="flex-1 px-4 py-2 text-sm font-medium rounded-lg transition-colors"
                  style={
                    mode === 'blur'
                      ? { background: 'var(--color-accent)', color: 'var(--color-text-primary)' }
                      : { 
                          background: 'var(--color-bg-secondary)', 
                          color: 'var(--color-text-secondary)',
                          border: '1px solid var(--color-border)'
                        }
                  }
                >
                  Blur
                </button>
                <button
                  onClick={() => setMode('pixelate')}
                  className="flex-1 px-4 py-2 text-sm font-medium rounded-lg transition-colors"
                  style={
                    mode === 'pixelate'
                      ? { background: 'var(--color-accent)', color: 'var(--color-text-primary)' }
                      : { 
                          background: 'var(--color-bg-secondary)', 
                          color: 'var(--color-text-secondary)',
                          border: '1px solid var(--color-border)'
                        }
                  }
                >
                  Pixelate
                </button>
              </div>
            </div>

            {/* Confidence Threshold */}
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-primary)' }}>
                Confidence Threshold: {confidence.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                style={{
                  background: 'var(--color-bg-secondary)',
                  accentColor: 'var(--color-accent)'
                }}
              />
              <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
                <span>Less strict</span>
                <span>More strict</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div 
          className="px-6 py-3"
          style={{ 
            background: 'rgba(239, 68, 68, 0.1)', 
            borderBottom: '1px solid var(--color-border)' 
          }}
        >
          <p className="text-sm" style={{ color: 'var(--color-error)' }}>{error}</p>
        </div>
      )}

      {/* Image Comparison */}
      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Original */}
          <div>
            <h4 className="text-sm font-medium mb-2" style={{ color: 'var(--color-text-primary)' }}>
              Original
            </h4>
            <div 
              className="relative aspect-video rounded-lg overflow-hidden"
              style={{ background: 'var(--color-bg-tertiary)' }}
            >
              {originalUrl && (
                <img
                  src={originalUrl}
                  alt="Original"
                  className="w-full h-full object-contain"
                />
              )}
            </div>
          </div>

          {/* Redacted */}
          <div>
            <h4 className="text-sm font-medium mb-2" style={{ color: 'var(--color-text-primary)' }}>
              Redacted
            </h4>
            <div 
              className="relative aspect-video rounded-lg overflow-hidden"
              style={{ background: 'var(--color-bg-tertiary)' }}
            >
              {isProcessing && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="flex flex-col items-center gap-3">
                    <div 
                      className="animate-spin rounded-full h-12 w-12 border-b-2"
                      style={{ borderColor: 'var(--color-accent)' }}
                    ></div>
                    <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                      Processing...
                    </p>
                  </div>
                </div>
              )}
              {redactedUrl && !isProcessing && (
                <img
                  src={redactedUrl}
                  alt="Redacted"
                  className="w-full h-full object-contain"
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


