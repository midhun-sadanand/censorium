# Example Images

This directory contains sample images for testing Censorium.

## Usage

Place test images here to try out the system:

```bash
# Process single image
cd ../backend
source venv/bin/activate
python run_redaction.py --input ../examples/test.jpg --output ../examples/output/redacted.jpg

# Process all images in this directory
python run_redaction.py --input ../examples --output ../examples/output
```

## Recommended Test Images

For comprehensive testing, include images with:

1. **Faces**:
   - Single face (frontal)
   - Multiple faces
   - Profile views
   - Partially occluded faces
   - Various lighting conditions

2. **License Plates**:
   - Clear, frontal plates
   - Angled plates
   - Multiple plates in scene
   - Different countries/regions

3. **Edge Cases**:
   - Very small faces (<20px)
   - High resolution (4K+)
   - Low resolution (<480p)
   - High noise/blur
   - Extreme lighting

## Sample Image Sources (Public Domain)

- **Unsplash**: https://unsplash.com/ (free high-quality images)
- **Pexels**: https://www.pexels.com/ (free stock photos)
- **Pixabay**: https://pixabay.com/ (free images and videos)

## Dataset Links (for Evaluation)

- **WIDER FACE**: http://shuoyang1213.me/WIDERFACE/
- **CCPD**: https://github.com/detectRecog/CCPD
- **CelebA**: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

---

Note: Always respect copyright and privacy when using test images. Use public domain or appropriately licensed images only.




