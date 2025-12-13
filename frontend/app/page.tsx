'use client';

import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import User from '@geist-ui/icons/user';
import Lock from '@geist-ui/icons/lock';
import Zap from '@geist-ui/icons/zap';
import RedactionViewer from '@/components/RedactionViewer';
import Sidebar from '@/components/Sidebar';
import { checkHealth } from '@/lib/api';

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  // Check API health on mount
  useEffect(() => {
    checkHealth()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'));
  }, []);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(prevFiles => [...prevFiles, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp']
    },
    multiple: true
  });

  const removeFile = (index: number) => {
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  const clearAll = () => {
    setFiles([]);
  };

  return (
    <div className="min-h-screen flex" style={{ background: 'var(--color-bg-primary)' }}>
      <Sidebar apiStatus={apiStatus} />

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        {/* Header */}
        <header className="px-8 py-6" style={{ borderBottom: '1px solid var(--color-border)' }}>
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl" style={{ color: 'var(--color-text-primary)' }}>
                Image Redaction
              </h2>
              <p className="mt-1 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                Upload images to automatically detect and redact faces and license plates
              </p>
            </div>
            {files.length > 0 && (
              <div className="flex items-center gap-3">
                <button
                  onClick={clearAll}
                  className="px-4 py-2 text-sm font-medium rounded-lg transition-colors"
                  style={{ 
                    background: 'var(--color-bg-secondary)', 
                    color: 'var(--color-text-secondary)',
                    border: '1px solid var(--color-border)'
                  }}
                >
                  Clear All
                </button>
                <div {...getRootProps()}>
                  <input {...getInputProps()} />
                  <button
                    className="px-4 py-2 text-sm font-medium rounded-lg transition-colors"
                    style={{ 
                      background: 'var(--color-accent)', 
                      color: 'var(--color-text-primary)'
                    }}
                  >
                    Add Images
                  </button>
                </div>
              </div>
            )}
          </div>
        </header>

        <div className="p-8">
          {/* Upload Zone */}
          {files.length === 0 && (
            <div className="mb-8">
              <div
                {...getRootProps()}
                className="rounded-lg p-16 text-center cursor-pointer transition-all"
                style={{
                  border: `2px dashed ${isDragActive ? 'var(--color-accent)' : 'var(--color-border)'}`,
                  background: isDragActive ? 'rgba(74, 144, 226, 0.05)' : 'var(--color-bg-secondary)'
                }}
              >
                <input {...getInputProps()} />
                <svg
                  className="mx-auto h-16 w-16 mb-4"
                  style={{ color: 'var(--color-text-muted)' }}
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                >
                  <path
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <p className="text-lg font-medium mb-2" style={{ color: 'var(--color-text-primary)' }}>
                  {isDragActive ? 'Drop files here' : 'Drag & drop images here'}
                </p>
                <p className="text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
                  or click to select files
                </p>
                <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                  Supports .jpeg, .png, .bpm
                </p>
              </div>
            </div>
          )}

          {/* Files Grid */}
          {files.length > 0 && (
            <div>
              <div className="mb-6">
                <h3 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>
                  {files.length} {files.length === 1 ? 'Image' : 'Images'} Loaded
                </h3>
              </div>

              <div className="space-y-6">
                {files.map((file, index) => (
                  <RedactionViewer
                    key={`${file.name}-${index}`}
                    file={file}
                    onRemove={() => removeFile(index)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Info Cards */}
          {files.length === 0 && (
            <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
              <div 
                className="rounded-lg p-6"
                style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
              >
                <div 
                  className="w-12 h-12 rounded-lg flex items-center justify-center mb-4"
                  style={{ background: 'var(--color-bg-tertiary)', color: 'var(--color-accent)' }}
                >
                  <User size={24} />
                </div>
                <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                  Face Detection
                </h3>
                <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                  Automatically detects and redacts faces using advanced neural networks
                </p>
              </div>

              <div 
                className="rounded-lg p-6"
                style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
              >
                <div 
                  className="w-12 h-12 rounded-lg flex items-center justify-center mb-4"
                  style={{ background: 'var(--color-bg-tertiary)', color: 'var(--color-accent)' }}
                >
                  <Lock size={24} />
                </div>
                <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                  License Plates
                </h3>
                <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                  Identifies and obscures license plates to protect privacy
                </p>
              </div>

              <div 
                className="rounded-lg p-6"
                style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
              >
                <div 
                  className="w-12 h-12 rounded-lg flex items-center justify-center mb-4"
                  style={{ background: 'var(--color-bg-tertiary)', color: 'var(--color-accent)' }}
                >
                  <Zap size={24} />
                </div>
                <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                  Real-time Processing
                </h3>
                <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                  Fast inference optimized for modern hardware
                </p>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
