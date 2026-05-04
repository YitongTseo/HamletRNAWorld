#version 330

in vec2 v_world;

uniform vec2 u_portal;       // world position of the portal mouth
uniform float u_portal_r;    // hole radius in world units
uniform vec3 u_pipe_color;
uniform vec2 u_pipe_x;       // (x_min, x_max) of the pipe rectangle
uniform vec2 u_pipe_y;       // (y_min, y_max) of the pipe rectangle

out vec4 frag_color;

void main() {
    if (v_world.x < u_pipe_x.x || v_world.x > u_pipe_x.y) discard;
    if (v_world.y < u_pipe_y.x || v_world.y > u_pipe_y.y) discard;
    float d = length(v_world - u_portal);
    if (d < u_portal_r) discard;
    // Soft inner edge so the hole has a gradient rim.
    float edge = smoothstep(u_portal_r, u_portal_r + 0.6, d);
    frag_color = vec4(u_pipe_color * edge, 1.0);
}
