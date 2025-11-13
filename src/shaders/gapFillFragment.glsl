/**
 * SDF-Splats Gap Filling Post-Processing Fragment Shader
 *
 * This shader performs gap detection and filling as a post-processing pass.
 * It takes the rendered splat image as input and fills detected gaps using
 * sphere tracing through the Gaussian density field.
 */

precision highp float;
precision highp int;

// Input textures
uniform sampler2D splatRenderTexture;  // Rendered splat image
uniform sampler2D splatDataTexture;    // Packed splat data (positions, etc.)
uniform sampler2D splatColorTexture;   // Splat colors
uniform sampler2D depthTexture;        // Depth buffer

// Screen parameters
uniform vec2 resolution;               // Screen resolution
uniform vec2 texelSize;                // 1.0 / resolution

// Camera parameters
uniform vec3 cameraPosition;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 inverseProjectionMatrix;
uniform mat4 inverseViewMatrix;
uniform float cameraNear;
uniform float cameraFar;

// SDF Gap-filling parameters
uniform bool sdfEnabled;
uniform float sdfAlphaThreshold;       // Gap detection threshold
uniform float sdfGradientThreshold;    // Edge detection threshold
uniform float sdfDensityThreshold;     // SDF iso-surface threshold
uniform int sdfMaxSteps;               // Max raymarch steps
uniform float sdfMaxDistance;          // Max raymarch distance
uniform float sdfSafetyFactor;         // Sphere trace safety (0.5-1.0)
uniform float sdfSurfaceEpsilon;       // Surface hit epsilon
uniform float sdfRaymarchRadius;       // Gaussian sampling radius
uniform bool sdfDebugMode;             // Debug visualization

// Splat data
uniform int splatCount;                // Total number of splats
uniform vec3 sceneBoundsMin;           // Scene bounding box
uniform vec3 sceneBoundsMax;

// Output
out vec4 fragColor;

// Input from vertex shader
in vec2 vUv;

/**
 * Unpack world position from screen UV and depth
 */
vec3 getWorldPosition(vec2 uv, float depth) {
    // Convert to NDC
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);

    // To view space
    vec4 viewPos = inverseProjectionMatrix * ndc;
    viewPos /= viewPos.w;

    // To world space
    vec4 worldPos = inverseViewMatrix * viewPos;
    return worldPos.xyz;
}

/**
 * Get world-space ray direction from screen UV
 */
vec3 getRayDirection(vec2 uv) {
    // NDC coordinates
    vec4 clipCoords = vec4(uv * 2.0 - 1.0, -1.0, 1.0);

    // View space
    vec4 viewCoords = inverseProjectionMatrix * clipCoords;
    viewCoords = vec4(viewCoords.xy, -1.0, 0.0);

    // World space
    vec3 worldDir = (inverseViewMatrix * viewCoords).xyz;
    return normalize(worldDir);
}

/**
 * Detect if current pixel is a gap that needs filling
 */
bool isGap(vec2 uv, float alpha) {
    // Check alpha threshold
    if (alpha >= sdfAlphaThreshold) {
        return false;
    }

    // Sample neighboring pixels for gradient computation
    float alphaR = texture(splatRenderTexture, uv + vec2(texelSize.x, 0.0)).a;
    float alphaL = texture(splatRenderTexture, uv - vec2(texelSize.x, 0.0)).a;
    float alphaU = texture(splatRenderTexture, uv + vec2(0.0, texelSize.y)).a;
    float alphaD = texture(splatRenderTexture, uv - vec2(0.0, texelSize.y)).a;

    // Compute gradient
    float gradX = abs(alphaR - alphaL);
    float gradY = abs(alphaU - alphaD);
    float gradient = max(gradX, gradY);

    // Gap detected at boundaries with low coverage
    return gradient > sdfGradientThreshold;
}

/**
 * Simplified splat data unpacking
 * Returns splat center position
 */
vec3 getSplatCenter(int splatIndex) {
    // Calculate texture coordinate for this splat
    // This is a simplified version - actual unpacking depends on data layout
    float u = float(splatIndex) / float(splatCount);
    vec4 data = texture(splatDataTexture, vec2(u, 0.5));
    return data.xyz;  // Assuming position is stored in xyz
}

/**
 * Get splat color
 */
vec4 getSplatColor(int splatIndex) {
    float u = float(splatIndex) / float(splatCount);
    return texture(splatColorTexture, vec2(u, 0.5));
}

/**
 * Sample Gaussian density field at a world position
 * Accumulates contributions from nearby splats
 */
float sampleDensity(vec3 worldPos) {
    float density = 0.0;
    float sigma = sdfRaymarchRadius * 0.35;  // Kernel standard deviation
    float sigma2 = sigma * sigma;
    float radius2 = sdfRaymarchRadius * sdfRaymarchRadius;

    // Sample a subset of splats for performance
    // In production, this would use a spatial grid acceleration structure
    int sampleStep = max(1, splatCount / 128);  // Sample up to 128 splats

    for (int i = 0; i < splatCount; i += sampleStep) {
        vec3 splatCenter = getSplatCenter(i);

        // Compute distance to splat
        vec3 diff = worldPos - splatCenter;
        float dist2 = dot(diff, diff);

        // Early rejection
        if (dist2 > radius2) {
            continue;
        }

        // Gaussian contribution (isotropic approximation)
        float contribution = exp(-dist2 / (2.0 * sigma2));
        density += contribution;
    }

    return density;
}

/**
 * Convert density to signed distance
 */
float densityToSDF(float density) {
    float diff = density - sdfDensityThreshold;

    // Inside surface (high density) = negative distance
    // Outside surface (low density) = positive distance
    if (density > sdfDensityThreshold) {
        return -abs(diff) * sdfRaymarchRadius;
    } else {
        return abs(diff) * sdfRaymarchRadius;
    }
}

/**
 * Compute SDF gradient for normal estimation using central differences
 */
vec3 computeNormal(vec3 p) {
    float eps = sdfSurfaceEpsilon * 10.0;

    float dX = sampleDensity(p + vec3(eps, 0.0, 0.0)) -
               sampleDensity(p - vec3(eps, 0.0, 0.0));
    float dY = sampleDensity(p + vec3(0.0, eps, 0.0)) -
               sampleDensity(p - vec3(0.0, eps, 0.0));
    float dZ = sampleDensity(p + vec3(0.0, 0.0, eps)) -
               sampleDensity(p - vec3(0.0, 0.0, eps));

    return normalize(vec3(dX, dY, dZ));
}

/**
 * Sample color from nearby splats at a world position
 */
vec3 sampleColor(vec3 worldPos) {
    vec3 color = vec3(0.0);
    float totalWeight = 0.0;
    float radius = sdfRaymarchRadius * 1.5;

    int sampleStep = max(1, splatCount / 64);

    for (int i = 0; i < splatCount; i += sampleStep) {
        vec3 splatCenter = getSplatCenter(i);
        float dist = length(worldPos - splatCenter);

        if (dist < radius) {
            float weight = 1.0 / (1.0 + dist);
            vec4 splatColor = getSplatColor(i);
            color += splatColor.rgb * weight;
            totalWeight += weight;
        }
    }

    if (totalWeight > 0.0) {
        color /= totalWeight;
    } else {
        color = vec3(0.5);  // Default gray
    }

    return color;
}

/**
 * Sphere tracing through the SDF
 * Returns: vec4(hitPos, hitFlag) where hitFlag = 1.0 for hit, 0.0 for miss
 */
vec4 sphereTrace(vec3 rayOrigin, vec3 rayDir) {
    float t = 0.01;  // Start offset

    for (int i = 0; i < sdfMaxSteps; i++) {
        vec3 p = rayOrigin + t * rayDir;

        // Sample density and convert to SDF
        float density = sampleDensity(p);
        float dist = densityToSDF(density);

        // Check for surface hit
        if (abs(dist) < sdfSurfaceEpsilon) {
            return vec4(p, 1.0);  // Hit!
        }

        // Check for ray escape
        if (t > sdfMaxDistance) {
            break;
        }

        // March forward with safety factor
        t += abs(dist) * sdfSafetyFactor;
    }

    return vec4(0.0, 0.0, 0.0, 0.0);  // Miss
}

/**
 * Apply simple lighting to filled surface
 */
vec3 applyLighting(vec3 baseColor, vec3 normal) {
    // Key light
    vec3 lightDir1 = normalize(vec3(0.5, 1.0, 0.3));
    float diffuse1 = max(0.0, dot(normal, lightDir1));

    // Fill light
    vec3 lightDir2 = normalize(vec3(-0.3, 0.2, -0.5));
    float diffuse2 = max(0.0, dot(normal, lightDir2)) * 0.5;

    // Ambient
    float ambient = 0.3;

    return baseColor * (ambient + diffuse1 * 0.6 + diffuse2 * 0.3);
}

void main() {
    // Sample the rendered splat image
    vec4 splatColor = texture(splatRenderTexture, vUv);

    // Early exit if gap filling is disabled
    if (!sdfEnabled) {
        fragColor = splatColor;
        return;
    }

    // Debug mode: visualize gaps in red
    if (sdfDebugMode) {
        // Simple debug: show alpha values
        // Red = gaps (low alpha)
        // Green = gradient boundaries
        // Original = well-covered areas

        float alpha = splatColor.a;

        // Sample neighbors for gradient
        float alphaR = texture(splatRenderTexture, vUv + vec2(texelSize.x, 0.0)).a;
        float alphaL = texture(splatRenderTexture, vUv - vec2(texelSize.x, 0.0)).a;
        float alphaU = texture(splatRenderTexture, vUv + vec2(0.0, texelSize.y)).a;
        float alphaD = texture(splatRenderTexture, vUv - vec2(0.0, texelSize.y)).a;

        float gradX = abs(alphaR - alphaL);
        float gradY = abs(alphaU - alphaD);
        float gradient = max(gradX, gradY);

        bool isLowAlpha = alpha < sdfAlphaThreshold;
        bool isHighGradient = gradient > sdfGradientThreshold;

        if (isLowAlpha && isHighGradient) {
            // Gap detected - show bright red
            fragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else if (isLowAlpha) {
            // Low alpha but no gradient - show dark red
            fragColor = vec4(0.5, 0.0, 0.0, 1.0);
        } else if (isHighGradient) {
            // High gradient at boundary - show green
            fragColor = vec4(0.0, 1.0, 0.0, 1.0);
        } else {
            // Well-covered - show original
            fragColor = splatColor;
        }
        return;
    }

    // Check if this pixel needs gap filling
    if (!isGap(vUv, splatColor.a)) {
        fragColor = splatColor;
        return;
    }

    // Perform gap filling via sphere tracing
    vec3 rayOrigin = cameraPosition;
    vec3 rayDir = getRayDirection(vUv);

    vec4 hitResult = sphereTrace(rayOrigin, rayDir);

    if (hitResult.w > 0.5) {
        // Surface hit - compute shading
        vec3 hitPos = hitResult.xyz;
        vec3 normal = computeNormal(hitPos);
        vec3 baseColor = sampleColor(hitPos);
        vec3 litColor = applyLighting(baseColor, normal);

        // Compute alpha from density
        float density = sampleDensity(hitPos);
        float hitAlpha = clamp(density / sdfDensityThreshold, 0.0, 1.0);

        // Blend with existing splat color (over operator)
        vec3 finalColor = splatColor.rgb + litColor * hitAlpha * (1.0 - splatColor.a);
        float finalAlpha = splatColor.a + hitAlpha * (1.0 - splatColor.a);

        fragColor = vec4(finalColor, finalAlpha);
    } else {
        // No hit - keep original
        fragColor = splatColor;
    }
}
