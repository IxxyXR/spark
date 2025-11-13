/**
 * Simple pass-through vertex shader for gap-filling post-processing
 */

precision highp float;

// Vertex attributes
in vec3 position;
in vec2 uv;

// Output to fragment shader
out vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}
