/**
 * API client for Censorium backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Detection {
  bbox: [number, number, number, number];
  confidence: number;
  entity_type: 'face' | 'license_plate';
}

export interface RedactionResponse {
  detections: Detection[];
  processing_time_ms: number;
  image_dimensions: [number, number];
}

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  version: string;
}

export interface RedactionOptions {
  mode?: 'blur' | 'pixelate';
  confidence_threshold?: number;
  padding_factor?: number;
  blur_kernel_size?: number;
  pixelate_block_size?: number;
}

/**
 * Check API health status
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error('API health check failed');
  }
  return response.json();
}

/**
 * Redact a single image
 */
export async function redactImage(
  file: File,
  options: RedactionOptions = {}
): Promise<Blob> {
  const formData = new FormData();
  formData.append('file', file);
  
  if (options.mode) formData.append('mode', options.mode);
  if (options.confidence_threshold !== undefined) {
    formData.append('confidence_threshold', options.confidence_threshold.toString());
  }
  if (options.padding_factor !== undefined) {
    formData.append('padding_factor', options.padding_factor.toString());
  }
  if (options.blur_kernel_size !== undefined) {
    formData.append('blur_kernel_size', options.blur_kernel_size.toString());
  }
  if (options.pixelate_block_size !== undefined) {
    formData.append('pixelate_block_size', options.pixelate_block_size.toString());
  }

  const response = await fetch(`${API_BASE_URL}/redact-image`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Redaction failed: ${error}`);
  }

  return response.blob();
}

/**
 * Get metadata for redacted image
 */
export async function getRedactionMetadata(
  file: File,
  options: RedactionOptions = {}
): Promise<RedactionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('return_metadata', 'true');
  
  if (options.mode) formData.append('mode', options.mode);
  if (options.confidence_threshold !== undefined) {
    formData.append('confidence_threshold', options.confidence_threshold.toString());
  }
  if (options.padding_factor !== undefined) {
    formData.append('padding_factor', options.padding_factor.toString());
  }

  const response = await fetch(`${API_BASE_URL}/redact-image`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Metadata fetch failed: ${error}`);
  }

  return response.json();
}

/**
 * Redact multiple images and get ZIP file
 */
export async function redactBatch(
  files: File[],
  options: RedactionOptions = {}
): Promise<Blob> {
  const formData = new FormData();
  
  files.forEach((file) => {
    formData.append('files', file);
  });
  
  if (options.mode) formData.append('mode', options.mode);
  if (options.confidence_threshold !== undefined) {
    formData.append('confidence_threshold', options.confidence_threshold.toString());
  }
  if (options.padding_factor !== undefined) {
    formData.append('padding_factor', options.padding_factor.toString());
  }
  if (options.blur_kernel_size !== undefined) {
    formData.append('blur_kernel_size', options.blur_kernel_size.toString());
  }
  if (options.pixelate_block_size !== undefined) {
    formData.append('pixelate_block_size', options.pixelate_block_size.toString());
  }

  const response = await fetch(`${API_BASE_URL}/redact-batch`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Batch redaction failed: ${error}`);
  }

  return response.blob();
}

/**
 * Preview detections without redaction
 */
export async function previewDetections(
  file: File,
  options: { confidence_threshold?: number; padding_factor?: number } = {}
): Promise<Blob> {
  const formData = new FormData();
  formData.append('file', file);
  
  if (options.confidence_threshold !== undefined) {
    formData.append('confidence_threshold', options.confidence_threshold.toString());
  }
  if (options.padding_factor !== undefined) {
    formData.append('padding_factor', options.padding_factor.toString());
  }

  const response = await fetch(`${API_BASE_URL}/preview-detections`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Preview failed: ${error}`);
  }

  return response.blob();
}

/**
 * Download blob as file
 */
export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}




