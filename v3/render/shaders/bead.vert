#version 330

in vec2 in_pos;
in vec3 in_color;     // per-bead RGB precomputed from embedding PCA

uniform mat4 u_proj;
uniform float u_point_size;

flat out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
    gl_PointSize = u_point_size;
}
