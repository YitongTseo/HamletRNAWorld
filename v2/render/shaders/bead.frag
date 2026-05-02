#version 330

flat in vec3 v_color;
out vec4 frag_color;

void main() {
    // gl_PointCoord is in [0,1] across the point sprite. Center it.
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d2 = dot(p, p);
    if (d2 > 1.0) discard;

    // Tight bright core + softer halo, both Gaussian-ish.
    float core = exp(-d2 * 6.0);
    float halo = exp(-d2 * 1.5) * 0.45;
    float intensity = core + halo;

    // Pre-multiplied; blends additively against black background.
    frag_color = vec4(v_color * intensity, intensity);
}
