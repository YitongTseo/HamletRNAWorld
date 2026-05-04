#version 330

flat in vec3 v_color;
out vec4 frag_color;

void main() {
    // Soft radial halo behind each letter — kept low-intensity so the glyph
    // rendered on top of it stays readable.
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d2 = dot(p, p);
    if (d2 > 1.0) discard;
    float halo = exp(-d2 * 2.2) * 0.40;
    frag_color = vec4(v_color * halo, halo);
}
