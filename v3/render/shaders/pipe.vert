#version 330

in vec2 in_pos;       // world-space corner of the pipe quad
uniform mat4 u_proj;
out vec2 v_world;

void main() {
    v_world = in_pos;
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
}
