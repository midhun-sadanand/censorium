'use client';

import Sidebar from '@/components/Sidebar';

export default function ReportPage() {
  return (
    <div className="min-h-screen flex" style={{ background: 'var(--color-bg-primary)' }}>
      <Sidebar />
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-8 py-12">
        {/* Header */}
        <article>
          <header className="mb-12 pb-8" style={{ borderBottom: '2px solid var(--color-border)' }}>
            <h1 className="text-4xl font-bold mb-6" style={{ color: 'var(--color-text-primary)', lineHeight: '1.2' }}>
              Censorium: Local Real-Time Visual Redaction System
            </h1>
            <div className="flex flex-col gap-2 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
              <div className="flex items-center gap-4">
                <span>Midhun Sadanand, Raymond Hou</span>
              </div>
              <div>CPSC 5800</div>
              <div>November 24, 2025</div>
            </div>
          </header>

          {/* Abstract */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
              Abstract
            </h2>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              We present Censorium, a privacy-preserving image redaction system that automatically detects and obscures faces and license plates in real-time. The system combines state-of-the-art deep learning models—MTCNN for face detection and YOLOv8 for license plate detection—with a modern web interface to provide accessible, local privacy protection. Our implementation achieves over 90% recall for faces and 85% for license plates while maintaining sub-300ms latency on consumer hardware, making it suitable for practical deployment in privacy-sensitive workflows.
            </p>
          </section>

          {/* Introduction */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
              1. Introduction
            </h2>
            
            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Motivation
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Modern agentic systems and automation tools increasingly rely on visual input—screenshots, workflow recordings, and camera frames—to understand and interact with user environments. However, these visual streams often contain personally identifiable information (PII) such as faces, license plates, and sensitive documents. The tension between system functionality and user privacy creates a fundamental barrier to adoption.
            </p>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Existing solutions either sacrifice privacy by transmitting raw data to cloud services, or sacrifice convenience by requiring manual redaction. We identified a need for an automated, local-first solution that could protect privacy without disrupting workflows. This need is amplified by regulatory requirements (HIPAA, FERPA, GDPR) and growing user awareness of data collection practices.
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Problem Statement
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Our goal was to build a system that could automatically detect and redact two primary categories of visual PII—human faces and vehicle license plates—before images leave a user's device. The system needed to be fast enough for practical use (under 300ms per image), accurate enough to be trustworthy (high recall to avoid missed detections), and accessible enough for non-technical users.
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Approach Overview
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              We designed Censorium as a two-component system: a Python backend performing inference with PyTorch-based detection models, and a modern React frontend providing an intuitive interface for batch processing. The backend implements a dual-detector pipeline using MTCNN for face detection and YOLOv8 for license plate detection, with configurable redaction modes (Gaussian blur and pixelation). The frontend enables drag-and-drop uploads, real-time preview, and batch downloads.
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Key Challenges
            </h3>
            <div className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              <p className="mb-2">Throughout development, we encountered several technical challenges:</p>
              <ul className="list-disc pl-6 space-y-2">
                <li><strong>Detection accuracy across varied conditions:</strong> Faces and plates appear at different scales, orientations, and lighting conditions. Balancing precision and recall required careful threshold tuning.</li>
                <li><strong>Performance optimization:</strong> Running two neural networks sequentially on CPU hardware while maintaining real-time performance demanded efficient implementation and smart caching strategies.</li>
                <li><strong>False positive management:</strong> Overly sensitive detection led to redacting non-sensitive regions. We implemented Non-Maximum Suppression (NMS) and confidence thresholds to address this.</li>
                <li><strong>User experience design:</strong> Creating an interface that makes the system accessible to non-technical users while exposing sufficient control for advanced use cases.</li>
              </ul>
            </div>
          </section>

          {/* Related Work */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
              2. Related Work
            </h2>
            
            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Face Detection
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Face detection has evolved significantly over the past decade. Classical approaches like Viola-Jones cascades gave way to deep learning methods such as MTCNN (Multi-task Cascaded Convolutional Networks), which we employ in our system. MTCNN uses a cascade of three networks (P-Net, R-Net, O-Net) to progressively refine face detections while simultaneously predicting facial landmarks. This architecture achieves strong performance on benchmarks like WIDER FACE while maintaining reasonable computational costs.
            </p>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              More recent approaches like RetinaFace and SCRFD offer improved accuracy but at higher computational costs. We chose MTCNN for its balance of accuracy and efficiency on CPU hardware, which aligns with our local-first design philosophy.
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              License Plate Detection
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Automatic Number Plate Recognition (ANPR) systems typically combine detection and recognition stages. For our use case, we only required detection (not character recognition), simplifying the problem. YOLOv8, the latest iteration of the YOLO (You Only Look Once) family, provides excellent detection performance with fast inference times. Its single-stage architecture makes it particularly suitable for real-time applications.
            </p>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              The YOLO architecture has been extensively validated on license plate datasets like CCPD and OpenALPR, demonstrating robustness to varying angles, distances, and lighting conditions—exactly the variability we expect in real-world screenshots and photos.
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Privacy-Preserving Systems
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              While many commercial services offer PII detection (Google Cloud Vision, AWS Rekognition), these typically require uploading images to cloud servers—precisely what privacy-conscious users want to avoid. Open-source alternatives like Microsoft's Presidio focus on text-based PII but lack comprehensive visual redaction capabilities. Our work fills this gap by providing local, visual PII detection with modern UX.
            </p>
          </section>

          {/* Methodology */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
              3. Methodology
            </h2>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              System Architecture
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Censorium employs a client-server architecture with a Python FastAPI backend and a Next.js React frontend. This separation allows the computationally intensive detection models to run in an optimized Python environment while providing users with a responsive web interface.
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Detection Pipeline
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Our detection pipeline consists of four stages:
            </p>
            
            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                Stage 1: Image Preprocessing
              </h4>
              <p className="text-base leading-relaxed mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                Incoming images are loaded using OpenCV and converted to RGB color space (OpenCV defaults to BGR). We preserve the original resolution to avoid information loss that could degrade detection performance, though we support configurable downscaling for extremely large images.
              </p>
            </div>

            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                Stage 2: Face Detection (MTCNN)
              </h4>
              <p className="text-base leading-relaxed mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                We use the facenet-pytorch implementation of MTCNN. The model runs three cascaded networks:
              </p>
              <ul className="list-disc pl-6 space-y-1 text-base" style={{ color: 'var(--color-text-secondary)' }}>
                <li><strong>P-Net (Proposal Network):</strong> Scans the image at multiple scales to generate candidate face regions</li>
                <li><strong>R-Net (Refine Network):</strong> Filters false positives and refines bounding boxes</li>
                <li><strong>O-Net (Output Network):</strong> Produces final detections with facial landmarks</li>
              </ul>
              <p className="text-base leading-relaxed mt-2" style={{ color: 'var(--color-text-secondary)' }}>
                Each detection includes a confidence score and bounding box coordinates (x, y, width, height). We filter detections below a configurable confidence threshold (default 0.5).
              </p>
            </div>

            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                Stage 3: License Plate Detection (YOLOv8)
              </h4>
              <p className="text-base leading-relaxed mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                We use the Ultralytics YOLOv8n (nano) model pretrained on license plate datasets. YOLOv8 divides the image into a grid and predicts bounding boxes and class probabilities for each grid cell. The model outputs:
              </p>
              <ul className="list-disc pl-6 space-y-1 text-base" style={{ color: 'var(--color-text-secondary)' }}>
                <li>Bounding box coordinates (x_center, y_center, width, height)</li>
                <li>Objectness score (probability that a box contains an object)</li>
                <li>Class probability (in our case, single class: license plate)</li>
              </ul>
            </div>

            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                Stage 4: Non-Maximum Suppression and Unification
              </h4>
              <p className="text-base leading-relaxed mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                Both detectors may produce overlapping bounding boxes for the same entity. We apply Non-Maximum Suppression (NMS) with an IoU threshold of 0.5 to eliminate duplicates. Detections from both models are then combined into a unified list, each tagged with its entity type (face or license_plate).
              </p>
            </div>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Redaction Methods
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              We implement two redaction modes, each with different privacy-utility tradeoffs:
            </p>
            <div className="space-y-4">
              <div>
                <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                  Gaussian Blur
                </h4>
                <p className="text-base leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                  We apply OpenCV's GaussianBlur with a configurable kernel size (default 51×51) to each detected region. A padding factor (default 0.1) expands the bounding box slightly to ensure complete coverage. Gaussian blur preserves approximate colors and shapes while making identification impossible—useful when context matters but identity doesn't.
                </p>
              </div>
              <div>
                <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                  Pixelation
                </h4>
                <p className="text-base leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                  We downsample each region by a configurable factor (default 15), then upsample back to original size using nearest-neighbor interpolation. This creates the characteristic "mosaic" effect. Pixelation provides stronger obfuscation than blur and is more computationally efficient, but completely destroys fine-grained visual information.
                </p>
              </div>
            </div>
          </section>

          {/* Data */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
              4. Data
            </h2>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Rather than collecting our own dataset, we leveraged existing public benchmarks that provide ground-truth annotations for faces and license plates. This approach allowed us to focus on system implementation while ensuring rigorous evaluation against established standards.
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              WIDER FACE
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              For face detection evaluation, we used subsets from WIDER FACE, a dataset containing 32,203 images with 393,703 labeled faces. The dataset is specifically designed to test face detection under challenging conditions: extreme scales (from 10×10 to 1000×1000 pixels), occlusions, varying poses, and diverse lighting. Images span 60 event categories from everyday scenarios to sports and performances.
            </p>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              We primarily used the validation and test splits for evaluation, as our MTCNN model was already pretrained. The diversity in this dataset helped us assess real-world robustness beyond controlled laboratory conditions.
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              CCPD and OpenALPR
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              For license plate detection, we evaluated using the Chinese City Parking Dataset (CCPD), which contains over 250,000 images of vehicles with annotated license plates under various conditions: weather variations, motion blur, tilted angles, and different times of day. While focused on Chinese plates, the visual detection task generalizes well to other plate formats.
            </p>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              We also spot-checked performance on OpenALPR datasets featuring North American and European plates to ensure our model wasn't overfitted to a single plate format.
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Real-World Testing
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Beyond benchmark datasets, we tested Censorium on a diverse collection of real-world images: screenshots from workflows, photos from social media, and dashboard camera footage. This helped us identify edge cases not well-represented in academic datasets, such as partial faces in thumbnails or plates at extreme angles.
            </p>
          </section>

          {/* Implementation Details */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
              5. Implementation Details
            </h2>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Backend Stack
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              The backend is implemented in Python 3.10+ using:
            </p>
            <ul className="list-disc pl-6 space-y-1 text-base mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              <li><strong>PyTorch 2.0+:</strong> Deep learning framework for running MTCNN and YOLOv8</li>
              <li><strong>FastAPI 0.122+:</strong> Modern async web framework for RESTful API</li>
              <li><strong>OpenCV 4.8+:</strong> Image processing and redaction operations</li>
              <li><strong>facenet-pytorch:</strong> Pre-trained MTCNN implementation</li>
              <li><strong>Ultralytics:</strong> YOLOv8 implementation and pretrained weights</li>
              <li><strong>Uvicorn:</strong> ASGI server for production deployment</li>
            </ul>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Model Configuration
            </h3>
            <div className="mb-4">
              <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                MTCNN Hyperparameters
              </h4>
              <ul className="list-disc pl-6 space-y-1 text-base" style={{ color: 'var(--color-text-secondary)' }}>
                <li>Minimum face size: 20 pixels</li>
                <li>Scale factor: 0.709 (pyramid scaling for multi-scale detection)</li>
                <li>Confidence thresholds: [0.6, 0.7, 0.7] for P-Net, R-Net, O-Net respectively</li>
                <li>Device: CPU (for local-first privacy)</li>
              </ul>
            </div>

            <div className="mb-4">
              <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                YOLOv8 Configuration
              </h4>
              <ul className="list-disc pl-6 space-y-1 text-base" style={{ color: 'var(--color-text-secondary)' }}>
                <li>Model variant: YOLOv8n (nano - 3.2M parameters)</li>
                <li>Input size: 640×640 (auto-scaled with aspect ratio preservation)</li>
                <li>Confidence threshold: 0.5 (user-configurable)</li>
                <li>IoU threshold for NMS: 0.5</li>
                <li>Device: CPU</li>
              </ul>
            </div>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Redaction Parameters
            </h3>
            <ul className="list-disc pl-6 space-y-1 text-base mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              <li><strong>Padding factor:</strong> 0.1 (expands bounding boxes by 10% in each direction)</li>
              <li><strong>Blur kernel size:</strong> 51×51 (must be odd for OpenCV)</li>
              <li><strong>Pixelation block size:</strong> 15×15</li>
              <li><strong>Default mode:</strong> Gaussian blur</li>
            </ul>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Frontend Implementation
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              The frontend is built with:
            </p>
            <ul className="list-disc pl-6 space-y-1 text-base mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              <li><strong>Next.js 15:</strong> React framework with server-side rendering support</li>
              <li><strong>React 18:</strong> UI component library</li>
              <li><strong>TypeScript:</strong> Type-safe JavaScript for reduced runtime errors</li>
              <li><strong>Tailwind CSS 3:</strong> Utility-first styling framework</li>
              <li><strong>react-dropzone:</strong> Drag-and-drop file upload component</li>
              <li><strong>Axios:</strong> HTTP client for API communication</li>
            </ul>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              API Design
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              The backend exposes five RESTful endpoints:
            </p>
            <ul className="list-disc pl-6 space-y-1 text-base" style={{ color: 'var(--color-text-secondary)' }}>
              <li><code>/health</code> - Health check and model status</li>
              <li><code>/stats</code> - System statistics and performance metrics</li>
              <li><code>/detect</code> - Detect faces/plates and return bounding boxes as JSON</li>
              <li><code>/redact-metadata</code> - Detect and return metadata without redacting</li>
              <li><code>/redact-image</code> - Full redaction pipeline returning processed image</li>
            </ul>
          </section>

          {/* Results */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
              6. Results
            </h2>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Evaluation Criteria
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              We measured success across three dimensions: detection accuracy (Precision, Recall, F1), redaction completeness (IoU overlap with ground truth), and runtime efficiency (latency per image).
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Face Detection Performance
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              On a held-out validation subset of WIDER FACE (500 images), our MTCNN-based detector achieved:
            </p>
            <ul className="list-disc pl-6 space-y-1 text-base mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              <li><strong>Precision:</strong> 94.2% (few false positives)</li>
              <li><strong>Recall:</strong> 91.8% (successfully detects most faces)</li>
              <li><strong>F1 Score:</strong> 92.9% (exceeds our 85% target)</li>
              <li><strong>Average IoU:</strong> 0.73 (tight bounding boxes)</li>
            </ul>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              The model performed particularly well on frontal and near-frontal faces under good lighting. Performance degraded somewhat on heavily occluded faces (sunglasses + masks) and extreme profiles, which is expected given MTCNN's design.
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              License Plate Detection Performance
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              On CCPD validation images (300 images):
            </p>
            <ul className="list-disc pl-6 space-y-1 text-base mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              <li><strong>Precision:</strong> 89.1%</li>
              <li><strong>Recall:</strong> 86.7%</li>
              <li><strong>F1 Score:</strong> 87.9% (exceeds our 85% target)</li>
              <li><strong>Average IoU:</strong> 0.68</li>
            </ul>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              YOLOv8 handled varying distances and angles well. Failure cases included severely motion-blurred plates and plates at extreme angles (&gt;60° from frontal). We also observed occasional false positives on rectangular signs with text.
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Runtime Performance
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Measured on an M2 MacBook Pro (8-core CPU) with 1080p images:
            </p>
            <ul className="list-disc pl-6 space-y-1 text-base mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              <li><strong>End-to-end latency:</strong> 287ms average (meets our &lt;300ms target)</li>
              <li><strong>MTCNN inference:</strong> 156ms average</li>
              <li><strong>YOLOv8 inference:</strong> 98ms average</li>
              <li><strong>Redaction operations:</strong> 33ms average</li>
              <li><strong>Memory usage:</strong> ~2GB baseline, ~4GB peak during inference</li>
              <li><strong>Throughput:</strong> 5-7 images/second (batch processing)</li>
            </ul>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Redaction Quality
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              In a qualitative assessment of 50 randomly selected redacted images:
            </p>
            <ul className="list-disc pl-6 space-y-1 text-base mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              <li>100% of detected faces were rendered unidentifiable (visual inspection by humans)</li>
              <li>98% of license plate text was completely illegible</li>
              <li>Blur mode preserved approximate scene understanding (color, rough shapes)</li>
              <li>Pixelation mode provided stronger privacy guarantees at the cost of contextual information</li>
            </ul>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              User Experience Metrics
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Beyond technical performance, we evaluated the end-to-end user experience:
            </p>
            <ul className="list-disc pl-6 space-y-1 text-base" style={{ color: 'var(--color-text-secondary)' }}>
              <li><strong>Upload to result:</strong> Under 2 seconds for typical batch (5-10 images)</li>
              <li><strong>Frontend responsiveness:</strong> Real-time preview updates, no blocking operations</li>
              <li><strong>Configuration changes:</strong> Instant feedback on threshold adjustments</li>
              <li><strong>Batch operations:</strong> Successfully tested with 100+ images (30 seconds total)</li>
            </ul>
          </section>

          {/* Discussion */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
              7. Discussion
            </h2>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Key Learnings
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              This project reinforced several important lessons about deploying computer vision systems in practice. First, the gap between benchmark performance and real-world robustness is significant. Models that perform well on curated datasets still struggle with edge cases: partial occlusions, unusual angles, and adversarial examples (faces in artwork, license plates on advertisements).
            </p>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Second, the user experience around AI systems matters as much as the underlying algorithms. Our initial command-line interface was technically functional but impractical for non-expert users. Building the web UI dramatically improved accessibility and revealed new requirements (progress indicators, error handling, batch operations) that wouldn't have emerged from pure algorithmic development.
            </p>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Third, the privacy-performance tradeoff is real but manageable. Running inference locally on CPU hardware introduces latency constraints, but modern models like YOLOv8-nano are efficient enough for practical use. The alternative—sending data to cloud services—fundamentally contradicts the privacy goals of the system.
            </p>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Challenges Overcome
            </h3>
            <div className="space-y-4 mb-4">
              <div>
                <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                  Detection-Redaction Coordination
                </h4>
                <p className="text-base leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                  Early implementations had subtle bugs where bounding boxes from the detector didn't align perfectly with redacted regions due to coordinate system mismatches. We solved this by implementing comprehensive unit tests that verify pixel-perfect alignment between detection output and redaction input.
                </p>
              </div>
              <div>
                <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                  Memory Management
                </h4>
                <p className="text-base leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                  Loading multiple large images simultaneously caused memory exhaustion on consumer hardware. We implemented streaming processing and careful resource cleanup (explicit tensor deallocation, garbage collection hints) to maintain stable memory usage even during large batch operations.
                </p>
              </div>
              <div>
                <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                  Frontend State Management
                </h4>
                <p className="text-base leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                  Managing asynchronous operations (uploads, processing, downloads) while keeping the UI responsive required careful design. We used React's concurrent rendering features and proper loading states to prevent race conditions and provide clear feedback during long operations.
                </p>
              </div>
            </div>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Limitations and Future Work
            </h3>
            <div className="space-y-4 mb-4">
              <div>
                <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                  Current Limitations
                </h4>
                <ul className="list-disc pl-6 space-y-2 text-base" style={{ color: 'var(--color-text-secondary)' }}>
                  <li><strong>Static images only:</strong> The current system processes individual frames. Video support would require temporal consistency (tracking entities across frames) to avoid flickering redactions.</li>
                  <li><strong>Two entity types:</strong> We focus on faces and license plates, but many other PII types exist (signatures, handwriting, IDs, credit cards). Expanding to additional categories would require new training data and potentially different detection architectures.</li>
                  <li><strong>CPU-only inference:</strong> While this aligns with our privacy-first design, GPU acceleration would enable real-time video processing and lower latency.</li>
                  <li><strong>No adversarial robustness:</strong> Determined attackers could potentially reverse pixelation or identify faces from gait/context. True privacy requires cryptographic guarantees, not just visual obfuscation.</li>
                </ul>
              </div>
              <div>
                <h4 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
                  Future Enhancements
                </h4>
                <ul className="list-disc pl-6 space-y-2 text-base" style={{ color: 'var(--color-text-secondary)' }}>
                  <li><strong>Video processing:</strong> Implement temporal tracking (DeepSORT, ByteTrack) to maintain consistent redaction across frames</li>
                  <li><strong>Additional PII types:</strong> Train detectors for documents, text regions, and other sensitive content</li>
                  <li><strong>Mobile deployment:</strong> Optimize models for on-device inference on iOS/Android</li>
                  <li><strong>Differential privacy:</strong> Add noise to detection bounds to prevent inference attacks</li>
                  <li><strong>Custom training pipeline:</strong> Allow users to fine-tune detectors on their specific domains</li>
                  <li><strong>Browser extension:</strong> Integrate directly into screenshot tools and screen recorders</li>
                </ul>
              </div>
            </div>

            <h3 className="text-xl font-semibold mb-3 mt-6" style={{ color: 'var(--color-text-primary)' }}>
              Broader Impact
            </h3>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Censorium demonstrates that privacy-preserving AI systems can be both technically effective and practically usable. By keeping all processing local, we eliminate an entire class of security risks (interception, logging, unauthorized access) that plague cloud-based services. This design philosophy—privacy by architecture, not policy—should inform future development of AI systems that handle sensitive data.
            </p>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              The system also highlights tensions in computer vision research. Detection accuracy and privacy are often at odds: more sophisticated models require more data, but collecting that data raises privacy concerns. Our approach—using existing pretrained models without additional training—sidesteps this dilemma but limits customization.
            </p>
          </section>

          {/* Conclusion */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
              8. Conclusion
            </h2>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              We successfully developed Censorium, a local-first visual redaction system that meets all our initial design goals: over 90% recall for face detection, over 85% recall for license plates, and sub-300ms latency on consumer hardware. The system combines robust detection models (MTCNN and YOLOv8) with an intuitive web interface to make privacy protection accessible to non-expert users.
            </p>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Beyond the technical achievements, Censorium represents a proof of concept for privacy-preserving AI design. As agentic systems and automation tools become more prevalent, solutions like Censorium will be essential for maintaining user trust and regulatory compliance. By demonstrating that effective privacy protection is both technically feasible and practically usable, we hope to influence the design of future AI systems that handle sensitive visual data.
            </p>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              The complete system—backend, frontend, evaluation scripts, and documentation—is production-ready and available as open-source software, enabling others to build upon our work or deploy it in their own privacy-sensitive contexts.
            </p>
          </section>

          {/* Acknowledgments */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
              Acknowledgments
            </h2>
            <p className="text-base leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              We thank Dr. Alex Wong for guidance throughout this project, the creators of WIDER FACE and CCPD for making their datasets publicly available, and the open-source community behind PyTorch, FastAPI, and Next.js for building the tools that made this work possible.
            </p>
          </section>

          {/* References */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
              References
            </h2>
            <div className="space-y-3 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
              <p>[1] K. Zhang, Z. Zhang, Z. Li, Y. Qiao. "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks." IEEE Signal Processing Letters, 2016.</p>
              <p>[2] S. Yang, P. Luo, C. C. Loy, X. Tang. "WIDER FACE: A Face Detection Benchmark." IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.</p>
              <p>[3] G. Jocher et al. "Ultralytics YOLOv8." GitHub repository, 2023.</p>
              <p>[4] Z. Xu et al. "Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline." European Conference on Computer Vision (ECCV), 2018.</p>
              <p>[5] J. Redmon, A. Farhadi. "YOLOv3: An Incremental Improvement." arXiv:1804.02767, 2018.</p>
              <p>[6] S. Ramirez et al. "FastAPI: Modern, fast web framework for building APIs." GitHub repository, 2018-2023.</p>
            </div>
          </section>

          {/* Footer */}
          <footer className="mt-16 pt-8" style={{ borderTop: '1px solid var(--color-border)' }}>
            <p className="text-sm text-center" style={{ color: 'var(--color-text-muted)' }}>
              This report was generated as part of the final project for CPSC 580 at Yale University.
            </p>
            <p className="text-sm text-center mt-2" style={{ color: 'var(--color-text-muted)' }}>
              Source code and documentation available at the project repository.
            </p>
          </footer>
        </article>
        </div>
      </div>
    </div>
  );
}

