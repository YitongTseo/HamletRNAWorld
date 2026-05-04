#version 330

// Per-vertex (4 vertices form one quad — TRIANGLE_STRIP):
in vec2 in_corner;          // ∈ [-0.5, +0.5] x [-0.5, +0.5]

// Per-instance (one per letter bead):
in vec2 in_pos;             // world position of the letter
in vec2 in_tangent;         // unit vector along the local backbone (right axis)
in vec2 in_uv_min;
in vec2 in_uv_max;
in vec3 in_color;

uniform mat4 u_proj;
uniform float u_glyph_size;

out vec2 v_uv;
flat out vec3 v_color;

void main() {
    // Build the letter's local frame: right = tangent, up = perpendicular CCW.
    vec2 right = in_tangent;
    vec2 up    = vec2(-in_tangent.y, in_tangent.x);
    vec2 offs  = (in_corner.x * right + in_corner.y * up) * u_glyph_size;
    vec2 world = in_pos + offs;
    gl_Position = u_proj * vec4(world, 0.0, 1.0);

    vec2 t = in_corner + 0.5;
    v_uv = mix(in_uv_min, in_uv_max, t);
    v_color = in_color;
}
