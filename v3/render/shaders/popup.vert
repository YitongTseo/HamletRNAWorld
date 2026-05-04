#version 330

// Position is already in NDC; UV is the standard [0,1] texture coord.
in vec2 in_pos;
in vec2 in_uv;

out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
