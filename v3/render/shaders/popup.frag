#version 330

in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 frag_color;

void main() {
    vec4 c = texture(u_tex, v_uv);
    if (c.a < 0.005) discard;
    // Standard alpha (not pre-multiplied). Caller switches to (SRC_ALPHA,
    // ONE_MINUS_SRC_ALPHA) blending for this pass.
    frag_color = c;
}
