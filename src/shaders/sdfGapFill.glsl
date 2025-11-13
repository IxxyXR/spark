/**
 * SDF-Splats: Hybrid Volumetric Gap Filling
 * GLSL Shader Functions for Real-Time Gap Filling
 *
 * This shader implements sphere tracing through a Signed Distance Field
 * constructed from the Gaussian splat density field to fill gaps in
 * under-sampled regions.
 */

#ifdef USE_SDF_GAP_FILL

// Uniforms for SDF gap filling
uniform bool sdfGapFillEnabled;
uniform float sdfAlphaThreshold;
uniform float sdfGradientThreshold;
uniform float sdfDensityThreshold;
uniform int sdfMaxSteps;
uniform float sdfMaxDistance;
uniform float sdfSafetyFactor;
uniform float sdfSurfaceEpsilon;
uniform float sdfRaymarchRadius;
uniform bool sdfDebug;

// Spatial grid uniforms
uniform sampler2D splatColor;  // Splat color/data texture
uniform sampler2D splatCenter; // Splat position texture
uniform sampler2D splatCov;    // Splat covariance texture
uniform usampler2D sdfGridTexture; // Spatial grid for acceleration
uniform vec3 sdfGridResolution;
uniform vec3 sdfGridBoundsMin;
uniform vec3 sdfGridBoundsMax;
uniform vec3 sdfGridCellSize;

// Camera uniforms (should be provided by main shader)
uniform vec3 cameraPosition;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

/**
 * Convert screen-space UV to world-space ray direction
 */
vec3 getWorldRayDirection(vec2 screenUV, mat4 invProjMat, mat4 invViewMat) {
    // NDC coordinates
    vec4 clipCoords = vec4(screenUV * 2.0 - 1.0, -1.0, 1.0);

    // View space
    vec4 viewCoords = invProjMat * clipCoords;
    viewCoords = vec4(viewCoords.xy, -1.0, 0.0);

    // World space
    vec3 worldDir = (invViewMat * viewCoords).xyz;
    return normalize(worldDir);
}

/**
 * Gap Detection: Determines if a pixel needs gap filling
 * Returns true if alpha is below threshold AND gradient is high (edge detection)
 */
bool isGap(float alpha, vec2 screenUV, vec2 texelSize) {
    // Check alpha threshold
    if (alpha >= sdfAlphaThreshold) {
        return false;
    }

    // Compute alpha gradient for edge detection
    float alphaRight = texture2D(splatColor, screenUV + vec2(texelSize.x, 0.0)).a;
    float alphaLeft = texture2D(splatColor, screenUV - vec2(texelSize.x, 0.0)).a;
    float alphaUp = texture2D(splatColor, screenUV + vec2(0.0, texelSize.y)).a;
    float alphaDown = texture2D(splatColor, screenUV - vec2(0.0, texelSize.y)).a;

    float gradX = abs(alphaRight - alphaLeft);
    float gradY = abs(alphaUp - alphaDown);
    float gradient = max(gradX, gradY);

    // Gap is detected at low alpha regions near boundaries
    return gradient > sdfGradientThreshold;
}

/**
 * Get grid cell indices for a world position
 */
ivec3 worldToGridCell(vec3 worldPos) {
    vec3 normalized = (worldPos - sdfGridBoundsMin) / (sdfGridBoundsMax - sdfGridBoundsMin);
    vec3 cellFloat = normalized * sdfGridResolution;
    return ivec3(clamp(cellFloat, vec3(0.0), sdfGridResolution - 1.0));
}

/**
 * Sample Gaussian field density at a point
 * Accumulates contributions from nearby Gaussians
 *
 * This is a simplified version that samples a local region.
 * For full implementation, we'd query the spatial grid for nearby splats.
 */
float sampleGaussianDensity(vec3 worldPos, sampler2D centerTex, sampler2D covTex, int splatCount) {
    float density = 0.0;
    float kernelRadius = sdfRaymarchRadius;
    float sigma = kernelRadius * 0.35; // Convert radius to standard deviation
    float sigma2 = sigma * sigma;

    // Get grid cell for spatial acceleration
    ivec3 gridCell = worldToGridCell(worldPos);

    // Sample nearby splats
    // In a full implementation, we'd query the spatial grid texture
    // to get only the splats in nearby cells

    // For now, we use a simplified uniform sampling approach
    // This should be replaced with proper spatial grid lookup

    int maxSamples = 32; // Limit samples for performance
    int step = max(1, splatCount / maxSamples);

    for (int i = 0; i < splatCount; i += step) {
        // Sample splat center from texture
        // Note: Texture coordinate calculation depends on data layout
        vec2 texCoord = vec2(float(i) / float(splatCount), 0.5);
        vec3 splatCenter = texture2D(centerTex, texCoord).xyz;

        // Distance to splat center
        vec3 diff = worldPos - splatCenter;
        float dist2 = dot(diff, diff);

        // Early rejection: skip if too far
        if (dist2 > kernelRadius * kernelRadius) {
            continue;
        }

        // Gaussian contribution (simplified isotropic version)
        // Full version would use covariance matrix from covTex
        float contribution = exp(-dist2 / (2.0 * sigma2));
        density += contribution;
    }

    return density;
}

/**
 * Convert density field to signed distance field
 * Negative = inside surface (high density)
 * Positive = outside surface (low density)
 * Zero = surface
 */
float densityToSDF(float density, float threshold) {
    // Distance approximation based on density difference from threshold
    float diff = density - threshold;

    // Simple linear mapping (can be improved with better distance metric)
    if (density > threshold) {
        // Inside surface
        return -abs(diff) * sdfRaymarchRadius;
    } else {
        // Outside surface
        return abs(diff) * sdfRaymarchRadius;
    }
}

/**
 * Compute SDF gradient for normal estimation
 * Uses central differences
 */
vec3 computeSDFGradient(vec3 p, sampler2D centerTex, sampler2D covTex, int splatCount) {
    float eps = sdfSurfaceEpsilon * 10.0; // Slightly larger epsilon for gradient

    float densityX1 = sampleGaussianDensity(p + vec3(eps, 0.0, 0.0), centerTex, covTex, splatCount);
    float densityX2 = sampleGaussianDensity(p - vec3(eps, 0.0, 0.0), centerTex, covTex, splatCount);
    float densityY1 = sampleGaussianDensity(p + vec3(0.0, eps, 0.0), centerTex, covTex, splatCount);
    float densityY2 = sampleGaussianDensity(p - vec3(0.0, eps, 0.0), centerTex, covTex, splatCount);
    float densityZ1 = sampleGaussianDensity(p + vec3(0.0, 0.0, eps), centerTex, covTex, splatCount);
    float densityZ2 = sampleGaussianDensity(p - vec3(0.0, 0.0, eps), centerTex, covTex, splatCount);

    vec3 gradient = vec3(
        densityX1 - densityX2,
        densityY1 - densityY2,
        densityZ1 - densityZ2
    );

    return normalize(gradient);
}

/**
 * Sample color from Gaussian field at a point
 * This is a simplified version - full implementation would properly
 * weight and blend nearby splat colors based on Gaussian kernels
 */
vec3 sampleGaussianColor(vec3 worldPos, vec3 normal, sampler2D centerTex, sampler2D colorTex, int splatCount) {
    vec3 color = vec3(0.5, 0.5, 0.5); // Default gray
    float totalWeight = 0.0;
    float kernelRadius = sdfRaymarchRadius * 1.5;

    int maxSamples = 16;
    int step = max(1, splatCount / maxSamples);

    for (int i = 0; i < splatCount; i += step) {
        vec2 texCoord = vec2(float(i) / float(splatCount), 0.5);
        vec3 splatCenter = texture2D(centerTex, texCoord).xyz;
        vec3 splatColor = texture2D(colorTex, texCoord).rgb;

        vec3 diff = worldPos - splatCenter;
        float dist = length(diff);

        if (dist < kernelRadius) {
            // Weight by inverse distance
            float weight = 1.0 / (1.0 + dist);
            color += splatColor * weight;
            totalWeight += weight;
        }
    }

    if (totalWeight > 0.0) {
        color /= totalWeight;
    }

    return color;
}

/**
 * Sphere Tracing Algorithm
 * Marches along ray using SDF to guide step size
 * Returns hit information: xyz = hit position, w = hit status (1.0 = hit, 0.0 = miss)
 */
vec4 sphereTrace(vec3 rayOrigin, vec3 rayDir, sampler2D centerTex, sampler2D covTex, int splatCount) {
    float t = 0.01; // Start slightly offset to avoid self-intersection
    vec3 hitPos = rayOrigin;

    for (int i = 0; i < SDF_MAX_STEPS; i++) {
        hitPos = rayOrigin + t * rayDir;

        // Sample Gaussian density field
        float density = sampleGaussianDensity(hitPos, centerTex, covTex, splatCount);

        // Convert to signed distance
        float dist = densityToSDF(density, sdfDensityThreshold);

        // Check for surface intersection
        if (abs(dist) < sdfSurfaceEpsilon) {
            return vec4(hitPos, 1.0); // Hit!
        }

        // Check for ray escape
        if (t > sdfMaxDistance) {
            break;
        }

        // March along ray (safety factor prevents over-stepping)
        t += abs(dist) * sdfSafetyFactor;
    }

    return vec4(0.0, 0.0, 0.0, 0.0); // Miss
}

/**
 * Main gap-filling function
 * Called from fragment shader for pixels that need filling
 */
vec4 fillGap(vec2 screenUV, vec4 splatColor, sampler2D centerTex, sampler2D covTex, int splatCount) {
    // Compute world-space ray
    mat4 invProj = inverse(projectionMatrix);
    mat4 invView = inverse(viewMatrix);
    vec3 rayDir = getWorldRayDirection(screenUV, invProj, invView);
    vec3 rayOrigin = cameraPosition;

    // Perform sphere tracing
    vec4 hitResult = sphereTrace(rayOrigin, rayDir, centerTex, covTex, splatCount);

    if (hitResult.w > 0.5) {
        // Hit surface - compute shading
        vec3 hitPos = hitResult.xyz;

        // Compute normal from SDF gradient
        vec3 normal = computeSDFGradient(hitPos, centerTex, covTex, splatCount);

        // Sample color from nearby splats
        vec3 hitColor = sampleGaussianColor(hitPos, normal, centerTex, splatColor, splatCount);

        // Simple lighting (Lambertian)
        vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
        float diffuse = max(0.0, dot(normal, lightDir));
        float ambient = 0.3;
        hitColor *= (ambient + (1.0 - ambient) * diffuse);

        // Compute alpha based on density
        float hitDensity = sampleGaussianDensity(hitPos, centerTex, covTex, splatCount);
        float hitAlpha = clamp(hitDensity / sdfDensityThreshold, 0.0, 1.0);

        // Blend with existing splat color (back-to-front blending)
        vec4 result = splatColor;
        result.rgb = result.rgb + hitColor * hitAlpha * (1.0 - result.a);
        result.a = result.a + hitAlpha * (1.0 - result.a);

        return result;
    }

    // No hit - return original splat color
    return splatColor;
}

/**
 * Debug visualization
 * Shows gaps in red for debugging
 */
vec4 debugGapVisualization(vec2 screenUV, vec4 splatColor, vec2 texelSize) {
    if (isGap(splatColor.a, screenUV, texelSize)) {
        return vec4(1.0, 0.0, 0.0, 1.0); // Red for gaps
    }
    return splatColor;
}

#endif // USE_SDF_GAP_FILL
