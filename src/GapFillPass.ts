import * as THREE from 'three';
import { PackedSplats } from './PackedSplats';
import { SDFGapFiller, SDFGapFillerParams } from './SDFGapFiller';

// Import shader code
import gapFillVertexShader from './shaders/gapFillVertex.glsl';
import gapFillFragmentShader from './shaders/gapFillFragment.glsl';

/**
 * GapFillPass: Post-processing pass for SDF-Splats gap filling
 *
 * This pass takes the rendered splat image and fills detected gaps
 * using sphere tracing through the Gaussian density field.
 *
 * Usage:
 * ```typescript
 * const gapFillPass = new GapFillPass(renderer, scene, camera);
 * gapFillPass.enabled = true;
 * gapFillPass.updateSplatData(packedSplats);
 *
 * // In render loop:
 * renderer.setRenderTarget(renderTarget);
 * renderer.render(scene, camera);
 * gapFillPass.render(renderer, renderTarget.texture);
 * ```
 */
export class GapFillPass {
    public enabled: boolean = false;

    private gapFiller: SDFGapFiller;
    private scene: THREE.Scene;
    private camera: THREE.OrthographicCamera;
    private material: THREE.ShaderMaterial;
    private quad: THREE.Mesh;
    private outputTarget: THREE.WebGLRenderTarget | null = null;

    // Splat data textures (to be provided externally)
    private splatDataTexture: THREE.Texture | null = null;
    private splatColorTexture: THREE.Texture | null = null;
    private packedSplats: PackedSplats | null = null;

    constructor(
        private renderer: THREE.WebGLRenderer,
        params?: SDFGapFillerParams
    ) {
        // Create gap filler with parameters
        this.gapFiller = new SDFGapFiller(params);

        // Create scene and camera for post-processing
        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

        // Create shader material
        this.material = this.createMaterial();

        // Create fullscreen quad
        const geometry = new THREE.PlaneGeometry(2, 2);
        this.quad = new THREE.Mesh(geometry, this.material);
        this.scene.add(this.quad);
    }

    /**
     * Create the gap-filling shader material
     */
    private createMaterial(): THREE.ShaderMaterial {
        return new THREE.ShaderMaterial({
            vertexShader: gapFillVertexShader,
            fragmentShader: gapFillFragmentShader,
            uniforms: {
                // Input textures
                splatRenderTexture: { value: null },
                splatDataTexture: { value: null },
                splatColorTexture: { value: null },
                depthTexture: { value: null },

                // Screen parameters
                resolution: { value: new THREE.Vector2() },
                texelSize: { value: new THREE.Vector2() },

                // Camera parameters
                cameraPosition: { value: new THREE.Vector3() },
                viewMatrix: { value: new THREE.Matrix4() },
                projectionMatrix: { value: new THREE.Matrix4() },
                inverseProjectionMatrix: { value: new THREE.Matrix4() },
                inverseViewMatrix: { value: new THREE.Matrix4() },
                cameraNear: { value: 0.1 },
                cameraFar: { value: 1000.0 },

                // SDF parameters (from gap filler)
                sdfEnabled: { value: false },
                sdfAlphaThreshold: { value: 0.2 },
                sdfGradientThreshold: { value: 0.05 },
                sdfDensityThreshold: { value: 0.5 },
                sdfMaxSteps: { value: 64 },
                sdfMaxDistance: { value: 10.0 },
                sdfSafetyFactor: { value: 0.8 },
                sdfSurfaceEpsilon: { value: 0.001 },
                sdfRaymarchRadius: { value: 0.5 },
                sdfDebugMode: { value: false },

                // Splat data
                splatCount: { value: 0 },
                sceneBoundsMin: { value: new THREE.Vector3(-10, -10, -10) },
                sceneBoundsMax: { value: new THREE.Vector3(10, 10, 10) },
            },
            depthTest: false,
            depthWrite: false,
        });
    }

    /**
     * Update gap filler parameters
     */
    updateParams(params: SDFGapFillerParams): void {
        this.gapFiller.updateParams(params);
        this.syncUniforms();
    }

    /**
     * Sync gap filler parameters to shader uniforms
     */
    private syncUniforms(): void {
        const uniforms = this.material.uniforms;
        uniforms.sdfEnabled.value = this.gapFiller.enabled;
        uniforms.sdfAlphaThreshold.value = this.gapFiller.alphaThreshold;
        uniforms.sdfGradientThreshold.value = this.gapFiller.gradientThreshold;
        uniforms.sdfDensityThreshold.value = this.gapFiller.densityThreshold;
        uniforms.sdfMaxSteps.value = this.gapFiller.maxSteps;
        uniforms.sdfMaxDistance.value = this.gapFiller.maxDistance;
        uniforms.sdfSafetyFactor.value = this.gapFiller.safetyFactor;
        uniforms.sdfSurfaceEpsilon.value = this.gapFiller.surfaceEpsilon;
        uniforms.sdfRaymarchRadius.value = this.gapFiller.raymarchRadius;
        uniforms.sdfDebugMode.value = this.gapFiller.debug;
    }

    /**
     * Update splat data for gap filling
     */
    updateSplatData(
        packedSplats: PackedSplats,
        dataTexture?: THREE.Texture,
        colorTexture?: THREE.Texture
    ): void {
        this.packedSplats = packedSplats;

        if (dataTexture) {
            this.splatDataTexture = dataTexture;
            this.material.uniforms.splatDataTexture.value = dataTexture;
        }

        if (colorTexture) {
            this.splatColorTexture = colorTexture;
            this.material.uniforms.splatColorTexture.value = colorTexture;
        }

        this.material.uniforms.splatCount.value = packedSplats.length;

        // Build spatial acceleration grid
        this.gapFiller.buildSpatialGrid(packedSplats);
    }

    /**
     * Set the output render target
     */
    setOutputTarget(target: THREE.WebGLRenderTarget | null): void {
        this.outputTarget = target;
    }

    /**
     * Render the gap-filling pass
     */
    render(
        camera: THREE.Camera,
        inputTexture: THREE.Texture,
        depthTexture?: THREE.Texture
    ): void {
        if (!this.enabled || !this.gapFiller.enabled) {
            // If disabled, just copy input to output
            if (this.outputTarget) {
                // Simple blit would go here
                // For now, skip rendering
            }
            return;
        }

        // Update uniforms
        const uniforms = this.material.uniforms;

        // Set input textures
        uniforms.splatRenderTexture.value = inputTexture;
        if (depthTexture) {
            uniforms.depthTexture.value = depthTexture;
        }

        // Update screen parameters
        const size = this.renderer.getSize(new THREE.Vector2());
        uniforms.resolution.value.copy(size);
        uniforms.texelSize.value.set(1.0 / size.x, 1.0 / size.y);

        // Update camera parameters
        if (camera instanceof THREE.PerspectiveCamera || camera instanceof THREE.OrthographicCamera) {
            uniforms.cameraPosition.value.copy(camera.position);
            uniforms.viewMatrix.value.copy(camera.matrixWorldInverse);
            uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
            uniforms.inverseProjectionMatrix.value.copy(camera.projectionMatrixInverse);
            uniforms.inverseViewMatrix.value.copy(camera.matrixWorld);

            if (camera instanceof THREE.PerspectiveCamera) {
                uniforms.cameraNear.value = camera.near;
                uniforms.cameraFar.value = camera.far;
            }
        }

        // Sync gap filler parameters
        this.syncUniforms();

        // Render to output target
        const currentTarget = this.renderer.getRenderTarget();
        this.renderer.setRenderTarget(this.outputTarget);
        this.renderer.render(this.scene, this.camera);
        this.renderer.setRenderTarget(currentTarget);
    }

    /**
     * Resize render targets
     */
    setSize(width: number, height: number): void {
        if (this.outputTarget) {
            this.outputTarget.setSize(width, height);
        }
    }

    /**
     * Clean up resources
     */
    dispose(): void {
        this.material.dispose();
        this.quad.geometry.dispose();
        this.gapFiller.dispose();

        if (this.outputTarget) {
            this.outputTarget.dispose();
        }
    }

    /**
     * Get the gap filler for direct parameter access
     */
    get gapFillerParams(): SDFGapFiller {
        return this.gapFiller;
    }
}
