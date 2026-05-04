#version 330

flat in vec3 v_color;
flat in int v_kind;
in float v_strain;
out vec4 frag_color;

uniform float u_intensity;
uniform int u_show_strain;

void main() {
    // In strain mode, hot bends pulse a touch brighter to draw the eye.
    float boost = (u_show_strain == 1 && v_kind == 0)
        ? 1.0 + 0.6 * clamp(v_strain * 0.5, 0.0, 1.0)
        : 1.0;
    float alpha = u_intensity * boost;
    frag_color = vec4(v_color * alpha, alpha);
}
