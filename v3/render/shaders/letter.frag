#version 330

in vec2 v_uv;
flat in vec3 v_color;
uniform sampler2D u_atlas;

out vec4 frag_color;

void main() {
    float a = texture(u_atlas, v_uv).r;     // luminance from grayscale atlas
    if (a < 0.02) discard;
    // Pre-multiplied for additive blending; the bright glyph tint shows on
    // top of the bead glow underneath.
    frag_color = vec4(v_color * a, a);
}
