#version 330

in vec2 in_pos;
in int in_kind;   // 0 = backbone, 1 = base pair

uniform mat4 u_proj;

flat out vec3 v_color;

void main() {
    vec3 backbone_color = vec3(0.55, 0.60, 0.75);
    vec3 bp_color       = vec3(1.00, 0.95, 0.80);
    v_color = (in_kind == 0) ? backbone_color : bp_color;
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
}
