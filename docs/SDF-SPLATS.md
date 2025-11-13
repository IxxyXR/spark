# SDF-Splats: Hybrid Volumetric Gap Filling for 3D Gaussian Splatting

## Overview

SDF-Splats is a novel hybrid rendering approach that combines explicit Gaussian Splat primitives with implicit Signed Distance Field (SDF) representations to address gap-filling in 3D Gaussian Splatting. This technique fills visible holes in under-sampled regions, transparency artifacts, and provides seamless reconstruction at real-time frame rates (30-60 FPS).

## The Gap-Filling Problem

Traditional 3D Gaussian Splatting excels at representing captured surfaces but suffers from several limitations:

- **Discrete Coverage**: Splats only exist where training images provided evidence
- **View-Dependent Holes**: Novel viewpoints expose gaps between captured splats
- **Under-Sampling Artifacts**: Thin structures and distant geometry exhibit visible discontinuities
- **Transparency Issues**: Alpha-blended regions can appear disconnected

## How It Works

### 1. Mathematical Foundation

SDF-Splats treats the Gaussian field as a continuous density function:

```
ρ(p) = Σ w_i · G_i(p)
```

Where:
- `p` = query point in 3D space
- `G_i(p)` = Gaussian contribution from splat i
- `w_i` = weighting factor

This density field is converted to a signed distance function:

```
d_SDF(p) = density > threshold ? -|density - threshold| : |density - threshold|
```

- Negative values = inside surface (high density)
- Positive values = outside surface (low density)
- Zero = surface

### 2. Gap Detection

Gaps are detected using dual criteria:

```
isGap = (alpha < alphaThreshold) AND (gradient(alpha) > gradientThreshold)
```

This ensures we only raymarch in genuine gaps at scene boundaries, avoiding wasted computation in empty backgrounds or well-covered regions.

### 3. Sphere Tracing

When a gap is detected, sphere tracing marches through the SDF:

```
1. Start at camera position
2. Sample density at current point
3. Convert density to SDF
4. If |SDF| < epsilon: surface hit!
5. Otherwise, march forward by |SDF| × safetyFactor
6. Repeat until hit or max distance reached
```

The SDF naturally guides step size—large steps in empty space, small steps near surfaces.

### 4. Surface Shading

When a surface is hit:
- Compute normal from SDF gradient using central differences
- Sample color from nearby splat colors weighted by distance
- Apply simple Lambertian lighting
- Blend result with existing splat color

## API Reference

### SDFGapFiller

The core class managing gap-filling parameters and spatial acceleration.

```typescript
import { SDFGapFiller } from 'spark';

const gapFiller = new SDFGapFiller({
    enabled: true,
    alphaThreshold: 0.2,        // Gap detection threshold (0.0-1.0)
    gradientThreshold: 0.05,    // Edge detection sensitivity
    densityThreshold: 0.5,      // SDF iso-surface threshold
    maxSteps: 64,               // Maximum raymarch iterations
    maxDistance: 10.0,          // Maximum raymarch distance
    safetyFactor: 0.8,          // Sphere trace safety (0.5-1.0)
    surfaceEpsilon: 0.001,      // Surface hit tolerance
    raymarchRadius: 0.5,        // Gaussian sampling radius
    gridResolution: 64,         // Spatial grid cells per axis
    debug: false                // Enable debug visualization
});
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | false | Enable/disable gap filling |
| `alphaThreshold` | number | 0.2 | Alpha below which gaps are detected (0.0-1.0) |
| `gradientThreshold` | number | 0.05 | Alpha gradient for edge detection |
| `densityThreshold` | number | 0.5 | Density threshold for SDF iso-surface |
| `maxSteps` | number | 64 | Maximum sphere tracing steps |
| `maxDistance` | number | 10.0 | Maximum raymarch distance |
| `safetyFactor` | number | 0.8 | Sphere tracing safety factor (0.5-1.0) |
| `surfaceEpsilon` | number | 0.001 | Surface detection epsilon |
| `raymarchRadius` | number | 0.5 | Radius for Gaussian field sampling |
| `gridResolution` | number | 64 | Spatial grid resolution (cells per axis) |
| `debug` | boolean | false | Enable debug visualization (shows gaps in red) |

**Methods:**

- `updateParams(params: SDFGapFillerParams)` - Update parameters
- `buildSpatialGrid(packedSplats: PackedSplats)` - Build spatial acceleration structure
- `getUniforms()` - Get shader uniforms for material integration
- `getShaderDefines()` - Get shader defines for conditional compilation
- `dispose()` - Clean up GPU resources

### GapFillPass

Post-processing pass for applying gap filling to rendered splat images.

```typescript
import { GapFillPass } from 'spark';

const gapFillPass = new GapFillPass(renderer, {
    enabled: true,
    alphaThreshold: 0.2,
    maxSteps: 64
});
```

**Methods:**

- `updateParams(params: SDFGapFillerParams)` - Update gap-filling parameters
- `updateSplatData(packedSplats, dataTexture?, colorTexture?)` - Provide splat data for gap filling
- `render(camera, inputTexture, depthTexture?)` - Execute the gap-filling pass
- `setOutputTarget(target)` - Set the output render target
- `setSize(width, height)` - Resize render targets
- `dispose()` - Clean up resources

## Usage Example

### Basic Integration

```typescript
import * as THREE from 'three';
import { SparkRenderer, GapFillPass } from 'spark';

// Create renderer
const renderer = new THREE.WebGLRenderer({ antialias: false });
const spark = new SparkRenderer({ renderer });

// Create gap-filling pass
const gapFillPass = new GapFillPass(renderer, {
    enabled: true,
    alphaThreshold: 0.25,
    maxSteps: 48,
    raymarchRadius: 0.4
});

// Create render targets
const renderTarget = new THREE.WebGLRenderTarget(
    window.innerWidth,
    window.innerHeight,
    {
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter
    }
);

const gapFillTarget = new THREE.WebGLRenderTarget(
    window.innerWidth,
    window.innerHeight,
    {
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter
    }
);

gapFillPass.setOutputTarget(gapFillTarget);

// Render loop
function animate() {
    requestAnimationFrame(animate);

    // 1. Render splats to render target
    renderer.setRenderTarget(renderTarget);
    renderer.render(scene, camera);

    // 2. Apply gap filling
    gapFillPass.render(camera, renderTarget.texture);

    // 3. Display result
    renderer.setRenderTarget(null);
    displayQuad.material.map = gapFillTarget.texture;
    renderer.render(displayScene, displayCamera);
}
```

### Advanced: Parameter Tuning

```typescript
// For sparse scenes with large gaps
gapFillPass.updateParams({
    alphaThreshold: 0.3,     // More aggressive gap detection
    maxSteps: 96,            // More raymarch steps
    raymarchRadius: 0.8,     // Larger sampling radius
    safetyFactor: 0.7        // Slower, safer marching
});

// For dense scenes (performance optimization)
gapFillPass.updateParams({
    alphaThreshold: 0.15,    // Conservative gap detection
    maxSteps: 32,            // Fewer raymarch steps
    raymarchRadius: 0.3,     // Smaller sampling radius
    safetyFactor: 0.9        // Faster marching
});

// Debug mode - visualize detected gaps
gapFillPass.updateParams({
    debug: true              // Gaps appear red
});
```

### Integration with SparkRenderer

```typescript
import { SparkRenderer, SplatMesh, GapFillPass } from 'spark';

// Load splat mesh
const splatMesh = new SplatMesh();
await splatMesh.load('model.ksplat');
scene.add(splatMesh);

// Update gap fill pass with splat data
const packedSplats = splatMesh.packedSplats;
if (packedSplats) {
    gapFillPass.updateSplatData(
        packedSplats,
        // Optional: provide data textures for more efficient sampling
        splatDataTexture,
        splatColorTexture
    );
}
```

## Performance Considerations

### GPU Performance

Gap filling is computationally expensive. Key factors:

- **maxSteps**: More steps = better quality but slower (32-96 recommended)
- **raymarchRadius**: Larger radius = more splat samples (0.3-0.8 recommended)
- **alphaThreshold**: Higher = fewer pixels processed (0.15-0.3 recommended)

### Spatial Acceleration

The spatial grid reduces raymarching cost from O(N·M) to O(K·M) where:
- N = total splat count
- M = raymarch steps
- K = average splats per grid cell (typically 10-50)

### Quality vs Performance Presets

**High Quality (30 FPS)**
```typescript
{
    maxSteps: 96,
    raymarchRadius: 0.8,
    alphaThreshold: 0.25,
    safetyFactor: 0.7
}
```

**Balanced (45-60 FPS)**
```typescript
{
    maxSteps: 64,
    raymarchRadius: 0.5,
    alphaThreshold: 0.2,
    safetyFactor: 0.8
}
```

**Performance (60+ FPS)**
```typescript
{
    maxSteps: 32,
    raymarchRadius: 0.3,
    alphaThreshold: 0.15,
    safetyFactor: 0.9
}
```

## Limitations

- **Simplified Density Sampling**: Current implementation uses isotropic Gaussian approximation. Full covariance matrix evaluation would improve accuracy but reduce performance.
- **Spatial Grid**: Currently uses uniform grid. Adaptive structures (octree/BVH) could improve efficiency in non-uniform scenes.
- **Color Sampling**: Uses weighted average of nearby splat colors. Doesn't account for view-dependent effects or spherical harmonics.
- **Lighting**: Simple Lambertian shading. Doesn't match complex lighting in original splat data.

## Future Enhancements

1. **Covariance-Aware Sampling**: Use full 3D Gaussian covariance for accurate density evaluation
2. **Adaptive Spatial Structures**: Octree or BVH for better performance in sparse scenes
3. **View-Dependent Effects**: Incorporate spherical harmonics for consistent appearance
4. **Temporal Coherence**: Reproject previous frame's gap-fill results to reduce flickering
5. **Neural Gap Filling**: Hybrid approach using lightweight neural network for color prediction

## References

- Kerbl et al. 2023. "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
- Hart, J. C. 1996. "Sphere tracing: A geometric method for the antialiased ray tracing of implicit surfaces"
- Quilez, I. "Distance Functions" - https://iquilezles.org/articles/distfunctions/

## Citation

If you use SDF-Splats in your research or project, please cite:

```bibtex
@misc{sdfsplats2024,
  title={SDF-Splats: Hybrid Volumetric Gap Filling for 3D Gaussian Splatting},
  author={Spark Team},
  year={2024},
  howpublished={\url{https://github.com/sparkjsdev/spark}}
}
```
