#version 330

in vec2 in_pos;
in int in_kind;        // 0 = backbone, 1 = base pair
in float in_strain;    // per-vertex backbone bending strain (0..2). 0 for bp endpoints.

uniform mat4 u_proj;
uniform int u_show_strain;   // 0 = normal, 1 = color backbone bonds by strain

flat out vec3 v_color;
flat out int v_kind;
out float v_strain;

vec3 strain_palette(float s) {
    // Smooth gradient: cool slate at zero strain, amber midway, hot red when bent past θ ≈ 90°.
    s = clamp(s, 0.0, 2.0);
    vec3 cool   = vec3(0.45, 0.55, 0.80);
    vec3 amber  = vec3(0.95, 0.75, 0.30);
    vec3 hot    = vec3(1.00, 0.25, 0.18);
    if (s < 0.5) return mix(cool, amber, s / 0.5);
    return mix(amber, hot, (s - 0.5) / 1.5);
}

void main() {
    vec3 backbone_color = vec3(0.55, 0.60, 0.75);
    vec3 bp_color       = vec3(1.00, 0.95, 0.80);

    if (in_kind == 1) {
        v_color = bp_color;
    } else if (u_show_strain == 1) {
        v_color = strain_palette(in_strain);
    } else {
        v_color = backbone_color;
    }
    v_kind = in_kind;
    v_strain = in_strain;
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
}
