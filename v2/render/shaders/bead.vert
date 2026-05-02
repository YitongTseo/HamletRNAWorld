#version 330

in vec2 in_pos;
in int in_base;

uniform mat4 u_proj;
uniform float u_point_size;

flat out vec3 v_color;

void main() {
    // Per-base colors. Order matches sequence.py: A, U, G, C.
    vec3 colors[4] = vec3[4](
        vec3(1.00, 0.65, 0.20),  // A — amber
        vec3(0.30, 0.85, 1.00),  // U — cyan
        vec3(1.00, 0.35, 0.85),  // G — magenta
        vec3(0.70, 1.00, 0.40)   // C — lime
    );
    v_color = colors[in_base];
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
    gl_PointSize = u_point_size;
}
